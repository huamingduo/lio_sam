#include "front_end.h"

int main(int argc, char **argv) {
  ROS_INFO("Start front_end_node");
  ros::init(argc, argv, "front_end_node");

  lio_sam::FrontEnd front_end;

  ros::MultiThreadedSpinner spinner(6);
  spinner.spin();

  return 0;
}