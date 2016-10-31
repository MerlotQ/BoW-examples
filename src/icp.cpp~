// use g2o ICP to aligin two point cloud
//#include <Eigen/StdVector>

#include <iostream>
#include <stdint.h>
#include<stdio.h>
#include<string>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include<pcl/common/transforms.h>
#include<pcl/visualization/cloud_viewer.h>

#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include <g2o/core/factory.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include "g2o/types/icp/types_icp.h"
//#include <g2o/types/slam3d/vertex_se3.h>


//using namespace Eigen;
using namespace std;
using namespace g2o;

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

void print4x4Matrix (const Eigen::Matrix4d & matrix);

int main()
{

  // The point clouds we will be using
    PointCloudT::Ptr cloud_in (new PointCloudT);  // Original point cloud
    PointCloudT::Ptr cloud_icp (new PointCloudT);  // ICP output point cloud
 
    string pcdfilepath("/home/bobo/code/g2o_exmaples/data/ism_test.pcd");
    if (pcl::io::loadPCDFile (pcdfilepath, *cloud_in) < 0)
      {
      cout<<"check the pcd file: "<<pcdfilepath<<endl;
        return (-1);
      }

    // Defining a rotation matrix and translation vector
   Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity ();

      // A rotation matrix 
      double theta = M_PI / 8;  // The angle of rotation in radians
      transformation_matrix (0, 0) = cos (theta);
      transformation_matrix (0, 1) = -sin (theta);
      transformation_matrix (1, 0) = sin (theta);
      transformation_matrix (1, 1) = cos (theta);

      // A translation on Z axis (0.4 meters)
      transformation_matrix (2, 3) = 0.4;
  // Display in terminal the transformation matrix
  std::cout << "Applying this rigid transformation to: cloud_in -> cloud_icp" << std::endl;
  print4x4Matrix (transformation_matrix);

  // Executing the transformation
   pcl::transformPointCloud (*cloud_in, *cloud_icp, transformation_matrix);

  SparseOptimizer optimizer;
  optimizer.setVerbose(false);

  g2o::BlockSolver_6_3::LinearSolverType* linearSolver = new  g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType> ();
  g2o::BlockSolver_6_3* block_solver = new g2o::BlockSolver_6_3( linearSolver );
  g2o::OptimizationAlgorithmLevenberg* solver= new g2o::OptimizationAlgorithmLevenberg( block_solver );

  optimizer.setAlgorithm(solver);

// set up two poses
//set up the first node
Eigen::Isometry3d cam;
cam =g2o::SE3Quat();
VertexSE3 *vc = new VertexSE3();
vc->setEstimate(cam);
vc->setId(0);
vc->setFixed(true);
optimizer.addVertex(vc);

cout<<"cam1"<<endl;
cout<<cam.matrix()<<endl;

//set up the second node
VertexSE3 *vc1= new VertexSE3();
Eigen::Isometry3d cam2;

for(int i=0;i<transformation_matrix.rows();i++)
  {
    for(int j=0;j<transformation_matrix.cols();j++)
      {
        cam2(i,j)=transformation_matrix(i,j);
      }
  }

cout<<"cam2"<<endl;
cout<<cam2.matrix()<<endl;

vc1->setEstimate(g2o::SE3Quat());
vc1->setId(1);
optimizer.addVertex(vc1);

  // set up point matches
  for (size_t i=0; i<cloud_in->points.size(); ++i)
  {
    Eigen::Vector3d pt0,pt1;//匹配点
    pt0=Eigen::Vector3d(cloud_in->points[i].x,cloud_in->points[i].y,cloud_in->points[i].z);
    //pt1=Eigen::Vector3d(cloud_icp->points[i].x,cloud_icp->points[i].y,cloud_icp->points[i].z) ;
    pt1= (cam2.inverse()) *pt0;
    
    // form edge, with normals in varioius positions，每个点的法向量
    Eigen::Vector3d nm0, nm1;
    nm0 << 0, i, 1;
    nm1 << 0, i, 1;
    nm0.normalize();
    nm1.normalize();

    Edge_V_V_GICP * e = new Edge_V_V_GICP();    // new edge with correct cohort for caching

    e->setVertex(0, dynamic_cast<VertexSE3*>(optimizer.vertex(0)));      // first viewpoint
    e->setVertex(1, dynamic_cast<VertexSE3*>(optimizer.vertex(1)));     // second viewpoint
    EdgeGICP meas;
    meas.pos0 = pt0;
    meas.pos1 = pt1;
    //meas.normal0 = nm0;
   // meas.normal1 = nm1;

    e->setMeasurement(meas);
    
    meas = e->measurement();
    // use this for point-plane
    //e->information() = meas.prec0(0.01);

    // use this for point-point 
    e->information().setIdentity();

    //e->setRobustKernel(true);
    //e->setHuberWidth(0.01);

    optimizer.addEdge(e);//

  }


  optimizer.initializeOptimization();
  optimizer.computeActiveErrors();
  cout << "Initial chi2 = " << FIXED(optimizer.chi2()) << endl;//chi2(),卡方分布

  optimizer.setVerbose(true);
  cout<<"optimizing ..."<<endl;
  optimizer.optimize(15);
  cout<<"done"<<endl;
  cout << dynamic_cast<VertexSE3*>(optimizer.vertex(0))->estimate().matrix()<<endl;
  cout << dynamic_cast<VertexSE3*>(optimizer.vertex(1))->estimate().matrix()<<endl;
  
}
void print4x4Matrix (const Eigen::Matrix4d & matrix)
{
  printf ("Rotation matrix :\n");
  printf ("    | %6.3f %6.3f %6.3f | \n", matrix (0, 0), matrix (0, 1), matrix (0, 2));
  printf ("R = | %6.3f %6.3f %6.3f | \n", matrix (1, 0), matrix (1, 1), matrix (1, 2));
  printf ("    | %6.3f %6.3f %6.3f | \n", matrix (2, 0), matrix (2, 1), matrix (2, 2));
  printf ("Translation vector :\n");
  printf ("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix (0, 3), matrix (1, 3), matrix (2, 3));
}
