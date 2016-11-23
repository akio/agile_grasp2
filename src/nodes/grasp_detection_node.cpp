#include <nodes/grasp_detection_node.h>
#include <sstream>
#include <tf2_eigen/tf2_eigen.h>


/** constants for input point cloud types */
const int GraspDetectionNode::PCD_FILE = 0; ///< *.pcd file
const int GraspDetectionNode::POINT_CLOUD_2 = 1; ///< sensor_msgs/PointCloud2
const int GraspDetectionNode::CLOUD_SIZED = 2; ///< agile_grasp2/CloudSized
const int GraspDetectionNode::CLOUD_INDEXED = 3; ///< agile_grasp2/CloudIndexed

/** constants for ROS service */
const int GraspDetectionNode::ALL_POINTS = 0; ///< service uses all points in the cloud
const int GraspDetectionNode::RADIUS = 1; ///< service uses all points within a radius given in the request
const int GraspDetectionNode::INDICES = 2; ///< service uses all points which are contained in an index list given in the request


GraspDetectionNode::GraspDetectionNode(ros::NodeHandle& node):
  has_cloud_(false),
  has_normals_(false),
  cloud_(new PointCloudRGBA),
  cloud_normals_(new PointCloudNormal),
  size_left_cloud_(0),
  has_samples_(true)
{
  node.param("use_importance_sampling", use_importance_sampling_, false);

  importance_sampling_ = new ImportanceSampling(node);
  grasp_detector_ = new GraspDetector(node);

  int cloud_type;
  node.param("cloud_type", cloud_type, POINT_CLOUD_2);

  node.param("hand_outer_diameter", outer_diameter_, 0.09);

  node.param("hand_depth", hand_depth_, 0.06);

  // read point cloud from ROS topic
  if (cloud_type != PCD_FILE)
  {
    std::string cloud_topic;
    node.param("cloud_topic", cloud_topic, std::string("/camera/depth_registered/points"));

    std::string samples_topic;
    node.param("samples_topic", samples_topic, std::string(""));

    // subscribe to input point cloud ROS topic
    if (cloud_type == POINT_CLOUD_2)
      cloud_sub_ = node.subscribe(cloud_topic, 1, &GraspDetectionNode::cloud_callback, this);
    else if (cloud_type == CLOUD_SIZED)
      cloud_sub_ = node.subscribe(cloud_topic, 1, &GraspDetectionNode::cloud_sized_callback, this);
    else if (cloud_type == CLOUD_INDEXED)
      cloud_sub_ = node.subscribe(cloud_topic, 1, &GraspDetectionNode::cloud_indexed_callback, this);

    // subscribe to input samples ROS topic
    if (!samples_topic.empty())
    {
      samples_sub_ = node.subscribe(samples_topic, 1, &GraspDetectionNode::samples_callback, this);
      has_samples_ = false;
      grasp_detector_->setUseIncomingSamples(true);
    }

    bool use_service;
    node.param("use_service", use_service, false);

    // uses a ROS service callback to provide grasps
    if (use_service)
    {
      grasps_service_ = node.advertiseService("find_grasps", &GraspDetectionNode::graspsServiceCallback, this);
    }
    // uses a ROS topic to publish grasps
    else
    {
      grasps_pub_ = node.advertise<agile_grasp2::GraspListMsg>("grasps", 10);
    }
    grasp_marker_pub_ = node.advertise<visualization_msgs::MarkerArray>("grasp_marker", 1);
    tf_broadcaster_.reset(new tf2_ros::TransformBroadcaster());
  }
}


void GraspDetectionNode::run()
{
  ros::Rate rate(1);
  ROS_INFO("Waiting for point cloud to arrive ...");

  while (ros::ok())
  {
    if (has_cloud_ && ((grasp_detector_->getUseIncomingSamples() && has_samples_) ||
                       !grasp_detector_->getUseIncomingSamples()))
    {
      ROS_INFO("Start processing...");
      // detect grasps in point cloud
      std::vector<GraspHypothesis> grasps = detectGraspPosesInTopic();


      // output grasps as ROS message
      agile_grasp2::GraspListMsg grasps_msg = createGraspListMsg(grasps);
      grasps_pub_.publish(grasps_msg);
      ROS_INFO_STREAM("Published " << grasps_msg.grasps.size() << " grasps.");

      geometry_msgs::TransformStamped t;
      visualization_msgs::MarkerArray marker_array;
      t.header.stamp = ros::Time::now();
      for (int i = 0; i < grasps_msg.grasps.size() ; ++i) {
        std::stringstream ss;
        const agile_grasp2::GraspMsg& g = grasps_msg.grasps[i];
        t.header.frame_id = grasps_msg.header.frame_id;
        ss << "grasp/" << i;
        t.child_frame_id = ss.str();
        ss.str(std::string());
        ss.clear();

        Eigen::Matrix4d transform;
        transform << g.approach.x, g.approach.y, g.approach.z, g.bottom.x,
                 g.axis.x, g.axis.y, g.axis.z, g.bottom.y,
                 g.binormal.x, g.binormal.y, g.binormal.z, g.bottom.z,
                 0, 0, 0, 1;

        t.transform = tf2::eigenToTransform(Eigen::Affine3d(transform)).transform;
        tf_broadcaster_->sendTransform(t);

        visualization_msgs::Marker marker;
        marker.header.frame_id = camera_frame_id_;
        marker.id = i;
        marker.action = visualization_msgs::Marker::ADD;
        marker.type = visualization_msgs::Marker::LINE_LIST;
        marker.lifetime.fromSec(10.0);
        marker.scale.x = 0.01;
        marker.scale.y = 0.0;
        marker.scale.z = 0.0;
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 0.8;

        double width = outer_diameter_;
        double hw = 0.5 * width;
        Eigen::Vector3d bottom(g.bottom.x, g.bottom.y, g.bottom.z);
        Eigen::Vector3d binormal(g.binormal.x, g.binormal.y, g.binormal.z);
        Eigen::Vector3d approach(g.approach.x, g.approach.y, g.approach.z);
        Eigen::Vector3d left_bottom = bottom + hw * binormal;
        Eigen::Vector3d right_bottom = bottom - hw * binormal;
        Eigen::Vector3d left_tip = left_bottom + hand_depth_ * approach;
        Eigen::Vector3d right_tip = right_bottom + hand_depth_ * approach;
        Eigen::Vector3d bottom_end = bottom - hand_depth_ * approach;

        geometry_msgs::Point lb;
        lb.x = left_bottom.x();
        lb.y = left_bottom.y();
        lb.z = left_bottom.z();
        geometry_msgs::Point rb;
        rb.x = right_bottom.x();
        rb.y = right_bottom.y();
        rb.z = right_bottom.z();
        geometry_msgs::Point lt;
        lt.x = left_tip.x();
        lt.y = left_tip.y();
        lt.z = left_tip.z();
        geometry_msgs::Point rt;
        rt.x = right_tip.x();
        rt.y = right_tip.y();
        rt.z = right_tip.z();
        geometry_msgs::Point bc;
        bc.x = bottom.x();
        bc.y = bottom.y();
        bc.z = bottom.z();
        geometry_msgs::Point be;
        be.x = bottom_end.x();
        be.y = bottom_end.y();
        be.z = bottom_end.z();

        marker.points.push_back(lb);
        marker.points.push_back(rb);

        marker.points.push_back(lb);
        marker.points.push_back(lt);

        marker.points.push_back(rb);
        marker.points.push_back(rt);

        marker.points.push_back(bc);
        marker.points.push_back(be);

        marker_array.markers.push_back(marker);
      }
      grasp_marker_pub_.publish(marker_array);


      // reset the system
      has_cloud_ = false;
      has_samples_ = false;
      ROS_INFO("Waiting for point cloud to arrive ...");
    }

    ros::spinOnce();
    rate.sleep();
  }
}


std::vector<GraspHypothesis> GraspDetectionNode::detectGraspPosesInFile(const std::string& file_name_left,
  const std::string& file_name_right)
{
  CloudCamera* cloud_cam;

  // load point cloud from file(s)
  if (file_name_right.length() == 0)
    cloud_cam = new CloudCamera(file_name_left);
  else
    cloud_cam = new CloudCamera(file_name_left, file_name_right);

  grasp_detector_->preprocessPointCloud(*cloud_cam);

  std::vector<GraspHypothesis> grasps;
  if (use_importance_sampling_)
    grasps = importance_sampling_->detectGraspPoses(*cloud_cam);
  else
    grasps = grasp_detector_->detectGraspPoses(*cloud_cam);

  delete cloud_cam;

  return grasps;
}


std::vector<GraspHypothesis> GraspDetectionNode::detectGraspPosesInTopic()
{
  ROS_INFO("detectGraspPosesInTopic");
  CloudCamera* cloud_cam;

  // cloud with surface normals
  if (has_normals_)
    cloud_cam = new CloudCamera(cloud_normals_, size_left_cloud_);
  // cloud without surface normals
  else
    cloud_cam = new CloudCamera(cloud_, size_left_cloud_);

  grasp_detector_->preprocessPointCloud(*cloud_cam);

  std::vector<GraspHypothesis> grasps;
  if (use_importance_sampling_) {
    grasps = importance_sampling_->detectGraspPoses(*cloud_cam);
  } else {
    grasps = grasp_detector_->detectGraspPoses(*cloud_cam);
  }

  delete cloud_cam;

  return grasps;
}


bool GraspDetectionNode::graspsServiceCallback(agile_grasp2::FindGrasps::Request& req,
                                               agile_grasp2::FindGrasps::Response& resp)
{
  ROS_INFO("Received grasp pose detection request ...");

  if (!has_cloud_)
  {
    ROS_INFO("No point cloud available!");
    return false;
  }

  CloudCamera* cloud_cam;

  // cloud with surface normals
  if (has_normals_)
    cloud_cam = new CloudCamera(cloud_normals_, size_left_cloud_);
  // cloud without surface normals
  else
    cloud_cam = new CloudCamera(cloud_, size_left_cloud_);

  // use all points in the point cloud
  if (req.grasps_signal == ALL_POINTS)
  {
    if (req.num_samples == 0)
      grasp_detector_->preprocessPointCloud(*cloud_cam);
    else
    {
      grasp_detector_->setNumSamples(req.num_samples);
      grasp_detector_->preprocessPointCloud(*cloud_cam);
    }
  }
  // use points within a given radius from a given center point in the point cloud
  else if (req.grasps_signal == RADIUS)
  {
    pcl::PointXYZRGBA centroid;
    centroid.x = req.centroid.x;
    centroid.y = req.centroid.y;
    centroid.z = req.centroid.z;
    std::vector<int> indices_ball = getSamplesInBall(cloud_cam->getCloudOriginal(), centroid, req.radius);
    cloud_cam->setSampleIndices(indices_ball);
  }
  // use points given by a list of indices into the point cloud
  else if (req.grasps_signal == INDICES)
  {
    std::vector<int> indices(req.indices.size());
    for(int i=0; i < req.indices.size(); i++)
      indices[i] = req.indices[i];
    cloud_cam->setSampleIndices(indices);
  }

  std::vector<GraspHypothesis> hands = grasp_detector_->detectGraspPoses(*cloud_cam);
  // TODO: fill response

  delete cloud_cam;

  return true;
}


std::vector<int> GraspDetectionNode::getSamplesInBall(const PointCloudRGBA::Ptr& cloud,
  const pcl::PointXYZRGBA& centroid, float radius)
{
  std::vector<int> indices;
  std::vector<float> dists;
  pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
  kdtree.setInputCloud(cloud);
  kdtree.radiusSearch(centroid, radius, indices, dists);
  return indices;
}


void GraspDetectionNode::cloud_callback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
  if (!has_cloud_)
  {
    if (msg->fields.size() == 6 && msg->fields[3].name == "normal_x" &&
        msg->fields[4].name == "normal_y" && msg->fields[5].name == "normal_z")
    {
      pcl::fromROSMsg(*msg, *cloud_normals_);
      has_normals_ = true;
      size_left_cloud_ = cloud_normals_->size();
      ROS_INFO_STREAM("Received cloud with " << cloud_normals_->points.size() << " points and their normals.");
    }
    else
    {
      pcl::fromROSMsg(*msg, *cloud_);
      size_left_cloud_ = cloud_->size();
      ROS_INFO_STREAM("Received cloud with " << cloud_->size() << " points.");
      camera_frame_id_ = msg->header.frame_id;
    }

    has_cloud_ = true;
  }
}


void GraspDetectionNode::cloud_sized_callback(const agile_grasp2::CloudSized& msg)
{
  if (!has_cloud_)
  {
    if (msg.cloud.fields.size() == 6 && msg.cloud.fields[3].name == "normal_x" &&
        msg.cloud.fields[4].name == "normal_y"
      && msg.cloud.fields[5].name == "normal_z")
    {
      pcl::fromROSMsg(msg.cloud, *cloud_normals_);
      has_normals_ = true;
    }
    else
    {
      pcl::fromROSMsg(msg.cloud, *cloud_);
    }

    size_left_cloud_ = msg.size_left.data;
    has_cloud_ = true;
    ROS_INFO_STREAM("Received cloud with " << cloud_->size() << " points (left: ) "
                    << size_left_cloud_ << ", right: "
      << cloud_->points.size() - size_left_cloud_);
  }
}


void GraspDetectionNode::cloud_indexed_callback(const agile_grasp2::CloudIndexed& msg)
{
  if (!has_cloud_)
  {
    if (msg.cloud.fields.size() == 6 && msg.cloud.fields[3].name == "normal_x" &&
        msg.cloud.fields[4].name == "normal_y"
      && msg.cloud.fields[3].name == "normal_x")
    {
      pcl::fromROSMsg(msg.cloud, *cloud_normals_);
      has_normals_ = true;
    }
    else
    {
      pcl::fromROSMsg(msg.cloud, *cloud_);
    }

    size_left_cloud_ = cloud_->size();
    grasp_detector_->setIndicesFromMsg(msg);
    has_cloud_ = true;
    ROS_INFO_STREAM("Received cloud with " << cloud_->size() << " points and "
                    << msg.indices.size() << " indices");
  }
}


void GraspDetectionNode::samples_callback(const agile_grasp2::SamplesMsg& msg)
{
  if (!has_samples_)
  {
    grasp_detector_->setSamplesMsg(msg);
    has_samples_ = true;
    ROS_INFO_STREAM("Received grasp samples message with " << msg.samples.size() << " samples");
  }
}


agile_grasp2::GraspListMsg GraspDetectionNode::createGraspListMsg(const std::vector<Handle>& handles)
{
  agile_grasp2::GraspListMsg msg;
  for (int i = 0; i < handles.size(); i++)
    msg.grasps.push_back(handles[i].convertToGraspMsg());
  msg.header.stamp = ros::Time::now();
  msg.header.frame_id = camera_frame_id_;
  return msg;
}


agile_grasp2::GraspListMsg GraspDetectionNode::createGraspListMsg(const std::vector<GraspHypothesis>& hands)
{
  agile_grasp2::GraspListMsg msg;
  for (int i = 0; i < hands.size(); i++)
    msg.grasps.push_back(hands[i].convertToGraspMsg());
  msg.header.stamp = ros::Time::now();
  msg.header.frame_id = camera_frame_id_;
  return msg;
}
