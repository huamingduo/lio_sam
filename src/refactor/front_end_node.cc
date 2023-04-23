#include "front_end.h"

int main(int argc, char **argv) {
  ros::init(argc, argv, "front_end_node");

  lio_sam::FrontEnd front_end;

  ros::MultiThreadedSpinner spinner(3);
  spinner.spin();

  return 0;
}