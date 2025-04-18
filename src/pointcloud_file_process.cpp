// ROS
#include <ros/ros.h>
#include <thread>
#include <iostream>

// PCL
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
using namespace std;

string file_name = string("intensity_sacn_all.pcd");
string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
std::string file_path = all_points_dir;

void visualize(const pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud);



//点云可视化
void visualize(const pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud){
    pcl::visualization::PCLVisualizer::Ptr viewer(boost::make_shared<pcl::visualization::PCLVisualizer>("point_cloud Viewer"));

    viewer->setBackgroundColor(255, 255, 255);
    //将坐标轴画进去
    // viewer->addCoordinateSystem(2); 
    // 设置相机位置      
    viewer->setCameraPosition(-15,0,0,1,0,0,0,0,1);
    //设置点云大小
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "point_cloud");  
    // 创建颜色处理对象，这里我们根据Z轴的值来着色
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> color_handler(point_cloud, "x");   
    // 加入点云
    // viewer->addPointCloud(point_cloud, "point_cloud");  //单色
    viewer->addPointCloud(point_cloud,color_handler, "point_cloud");  //带颜色


    while (!viewer->wasStopped())
    {
        viewer->spinOnce(10);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

int main(int argc, char **argv){
    ros::init(argc, argv, "pc_file_process");
    ros::NodeHandle nh("~");

    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_origin (boost::make_shared<pcl::PointCloud<pcl::PointXYZ> >());
    
    
    if(pcl::io::loadPCDFile<pcl::PointXYZ> (file_path, *point_cloud_origin) == -1){std::cout<< "打不开点云捏～"<<std::endl; return -1;}


    // 过滤自身周围的
    // pcl::CropBox<pcl::PointXYZ> crop_close;
    // crop_close.setMin(Eigen::Vector4f(-2.0, -4.0, -4.0, 1.0));
    // crop_close.setMax(Eigen::Vector4f( 2.0,  4.0,  4.0, 1.0));
    // crop_close.setInputCloud(point_cloud);
    // crop_close.setNegative(true);
    // crop_close.filter(*point_cloud);

    // pcl::VoxelGrid<pcl::PointXYZ> voxel;
    // voxel.setLeafSize(0.5,0.5,0.5);
    // voxel.setInputCloud(point_cloud_origin);
    // voxel.filter(*point_cloud_origin);

    visualize(point_cloud_origin);
    // ros::spinOnce();
    return 0;
}