#include "lidar_selection.h"

namespace lidar_selection {

LidarSelector::LidarSelector(const int gridsize, SparseMap* sparsemap ): grid_size(gridsize), sparse_map(sparsemap)
{
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    G = Matrix<double, DIM_STATE, DIM_STATE>::Zero();
    H_T_H = Matrix<double, DIM_STATE, DIM_STATE>::Zero();
    Rli = M3D::Identity();
    Rci = M3D::Identity();
    Rcw = M3D::Identity();
    Jdphi_dR = M3D::Identity();
    Jdp_dt = M3D::Identity();
    Jdp_dR = M3D::Identity();
    Pli = V3D::Zero();
    Pci = V3D::Zero();
    Pcw = V3D::Zero();
    width = 800;
    height = 600;
    kernel_dx_ = cv::Mat::zeros(1, 3, CV_32F);
    kernel_dy_ = cv::Mat::zeros(3, 1, CV_32F);
    kernel_dx_.at<float>(0, 0) = -0.5;
    kernel_dx_.at<float>(0, 2) = 0.5;
    kernel_dy_.at<float>(0, 0) = -0.5;
    kernel_dy_.at<float>(2, 0) = 0.5;
}

LidarSelector::~LidarSelector() 
{
    delete sparse_map;
    delete sub_sparse_map;
    delete[] grid_num;
    delete[] map_index;
    delete[] map_value;
    unordered_map<int, Warp*>().swap(Warp_map);
    unordered_map<VOXEL_KEY, float>().swap(sub_feat_map);
    unordered_map<VOXEL_KEY, VOXEL_POINTS*>().swap(feat_map);  
}

void LidarSelector::set_extrinsic(const V3D &transl, const M3D &rot)
{
    Pli = -rot.transpose() * transl;
    Rli = rot.transpose();
}

void LidarSelector::init()
{
    sub_sparse_map = new SubSparseMap;
    Rci = sparse_map->Rcl * Rli;
    Pci= sparse_map->Rcl*Pli + sparse_map->Pcl;
    M3D Ric;
    V3D Pic;
    Jdphi_dR = Rci;
    Pic = -Rci.transpose() * Pci;
    M3D tmp;
    tmp << SKEW_SYM_MATRX(Pic);
    Jdp_dR = -Rci * tmp;
    width = cam->width();
    height = cam->height();
    grid_n_width = static_cast<int>(width/grid_size);
    grid_n_height = static_cast<int>(height/grid_size);
    length = grid_n_width * grid_n_height;
    fx = cam->errorMultiplier2();
    fy = cam->errorMultiplier() / (4. * fx);
    grid_num = new int[length];
    grid_cnt = new int[length];
    map_index = new int[length];
    map_value = new float[length];
    map_dist = (float*)malloc(sizeof(float)*length);
    contri_values = (float*)malloc(sizeof(float)*length);
    memset(grid_num, TYPE_UNKNOWN, sizeof(int)*length);
    memset(grid_cnt, 0, sizeof(int)*length);
    memset(map_index, 0, sizeof(int)*length);
    memset(map_value, 0, sizeof(float)*length);
    voxel_points_.reserve(length);
    add_voxel_points_.reserve(length);
    patch_size_total = patch_size * patch_size;
    patch_size_half = static_cast<int>(patch_size/2);
    patch_cache.resize(patch_size_total);
    stage_ = STAGE_FIRST_FRAME;
    pg_down.reset(new PointCloudXYZI());
    weight_scale_ = 10;
    weight_function_.reset(new vk::robust_cost::HuberWeightFunction());
    // weight_function_.reset(new vk::robust_cost::TukeyWeightFunction());
    scale_estimator_.reset(new vk::robust_cost::UnitScaleEstimator());
    // scale_estimator_.reset(new vk::robust_cost::MADScaleEstimator());
}

void LidarSelector::loadParameters(ros::NodeHandle& nh) {
    nh.param<double>("fusion/grad_min", thr_gradmin_, 20);
    nh.param<int>("fusion/area_patch_size", area_patch_size, 5);
    nh.param<int>("fusion/n_features", n_features, 16);
    nh.param<int>("fusion/num_per_grid", num_per_grid, 2);
    nh.param<bool>("fusion/debug", debug_, false);
    nh.param<bool>("fusion/show_submap", show_submap_, false);
    nh.param<int>("fusion/add_submap_num", add_submap_num_, 100);
    nh.param<float>("fusion/img_scale", img_scale_, 2);
    pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/calib/err", 10);
    if (debug_) {
        moment_pub = nh.advertise<sensor_msgs::Image>("moment_image", 2000);
    }
}

void LidarSelector::reset_grid()
{
    memset(grid_num, TYPE_UNKNOWN, sizeof(int)*length);
    memset(map_index, 0, sizeof(int)*length);
    fill_n(map_dist, length, 10000);
    fill_n(contri_values, length, -1.0);
    std::vector<PointPtr>(length).swap(voxel_points_);
    std::vector<V3D>(length).swap(add_voxel_points_);
    voxel_points_.reserve(length);
    add_voxel_points_.reserve(length);
}

void LidarSelector::projectionJacobian(const V3D& p, MD(2,3)& du_dp) {
    M3D Rwi(state->rot_end);
    V3D Pwi(state->pos_end);
    M3D R_CW = Rci * Rwi.transpose();
    V3D P_CW = -Rci*Rwi.transpose()*Pwi + Pci;
    V3D pf = R_CW * p + P_CW;
    dpi(pf, du_dp);
}

void LidarSelector::dpi(V3D p, MD(2,3)& J) {
    const double x = p[0];
    const double y = p[1];
    const double z_inv = 1./p[2];
    const double z_inv_2 = z_inv * z_inv;
    J(0,0) = fx * z_inv;
    J(0,1) = 0.0;
    J(0,2) = -fx * x * z_inv_2;
    J(1,0) = 0.0;
    J(1,1) = fy * z_inv;
    J(1,2) = -fy * y * z_inv_2;
}

float LidarSelector::CheckGoodPoints(cv::Mat img, V2D uv)
{
    const float u_ref = uv[0];
    const float v_ref = uv[1];
    const int u_ref_i = floorf(uv[0]); 
    const int v_ref_i = floorf(uv[1]);
    const float subpix_u_ref = u_ref-u_ref_i;
    const float subpix_v_ref = v_ref-v_ref_i;
    uint8_t* img_ptr = (uint8_t*) img.data + (v_ref_i)*width + (u_ref_i);
    float gu = 2*(img_ptr[1] - img_ptr[-1]) + img_ptr[1-width] - img_ptr[-1-width] + img_ptr[1+width] - img_ptr[-1+width];
    float gv = 2*(img_ptr[width] - img_ptr[-width]) + img_ptr[width+1] - img_ptr[-width+1] + img_ptr[width-1] - img_ptr[-width-1];
    return fabs(gu)+fabs(gv);
}

void LidarSelector::getpatch(cv::Mat img, V2D pc, float* patch_tmp, int level) 
{
    const float u_ref = pc[0];
    const float v_ref = pc[1];
    const int scale =  (1<<level);
    const int u_ref_i = floorf(pc[0]/scale)*scale; 
    const int v_ref_i = floorf(pc[1]/scale)*scale;
    const float subpix_u_ref = (u_ref-u_ref_i)/scale;
    const float subpix_v_ref = (v_ref-v_ref_i)/scale;
    const float w_ref_tl = (1.0-subpix_u_ref) * (1.0-subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0-subpix_v_ref);
    const float w_ref_bl = (1.0-subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;
    for (int x=0; x<patch_size; x++) 
    {
        uint8_t* img_ptr = (uint8_t*) img.data + (v_ref_i-patch_size_half*scale+x*scale)*width + (u_ref_i-patch_size_half*scale);
        for (int y=0; y<patch_size; y++, img_ptr+=scale)
        {
            patch_tmp[patch_size_total*level+x*patch_size+y] = w_ref_tl*img_ptr[0] + w_ref_tr*img_ptr[scale] + w_ref_bl*img_ptr[scale*width] + w_ref_br*img_ptr[scale*width+scale];
        }
    }
}
//从图像中选取点加入是视觉全局地图
void LidarSelector::addSparseMap(cv::Mat img, PointCloudXYZI::Ptr pg, const std::vector<V3D>& V) 
{
    // double t0 = omp_get_wtime();
    reset_grid();

    // double t_b1 = omp_get_wtime() - t0;
    // t0 = omp_get_wtime();
    //计算梯度
    img_set.img_intensity = img.clone();
    if (blur_) {
        cv::Mat img_blur;
        cv::GaussianBlur(img_set.img_intensity, img_blur, cv::Size(3,3), 0);
        img_set.img_intensity = img_blur;
    }  
    cv::threshold(img_set.img_intensity, img_set.img_intensity, 255., 255., cv::THRESH_TRUNC);   

    // Convert to 8 bit for visualization
    img_set.img_intensity.convertTo(img_set.img_photo_u8, CV_8UC1, 1);

    // Calculate gradient images
    cv::filter2D(img_set.img_intensity, img_set.img_dx, CV_32F , kernel_dx_);
    cv::filter2D(img_set.img_intensity, img_set.img_dy, CV_32F , kernel_dy_);

    // Compute gradient image
    cv::Mat dx_abs, dy_abs, grad_img;
    cv::convertScaleAbs(img_set.img_dx, dx_abs);
    cv::convertScaleAbs(img_set.img_dy, dy_abs);
    cv::addWeighted(dx_abs, 0.5, dy_abs, 0.5, 0, grad_img);
    // Sort pixels by gradient score
    std::vector<std::pair<double, SuperPoint>> grad_scores_vec;
    grad_scores_vec.reserve(pg->size());
    //过滤梯度小的投影点
    for (int i=0; i<pg->size(); i++) 
    {
        V3D pt(pg->points[i].x, pg->points[i].y, pg->points[i].z);
        V2D pc(new_frame_->w2c(pt));
        if(new_frame_->cam_->isInFrame(pc.cast<int>(), (patch_size_half+1)*8)) // 20px is the patch size in the matcher
        {
            size_t v = static_cast<size_t>(pc[1]); // v is the y coordinate of the point
            size_t u = static_cast<size_t>(pc[0]); // u is the x coordinate of the point
            if(grad_img.ptr<uchar>(v)[u] > thr_gradmin_)
            {
                grad_scores_vec.push_back(std::make_pair(grad_img.ptr<uchar>(v)[u], SuperPoint(pt, cv::Point(u, v))));
            }
        }
    }
    // ROS_ERROR("grad_scores_vec size: %d, thr_gradmid_: %f", grad_scores_vec.size(), thr_gradmin_);
    //按照梯度大小排序
    std::sort( grad_scores_vec.begin(), grad_scores_vec.end(),
        [&]( const std::pair<double, SuperPoint>& lhs, const std::pair<double, SuperPoint>& rhs )
        {
            return lhs.first > rhs.first;
        } );
    //排序后加加入候选点集合
    std::vector<SuperPoint> candidates;
    for (auto& score_pair : grad_scores_vec) {
        candidates.push_back(score_pair.second);
    }

    // Vector to store scores for each direction
    std::vector<std::vector<std::pair<double, int>>> scores_vec(V.size(), std::vector<std::pair<double, int>>());
    for (size_t vec_idx = 0u; vec_idx < V.size(); vec_idx++) {
        scores_vec[vec_idx] = std::vector<std::pair<double, int>>(candidates.size(), std::make_pair(0, 0));
    }
    //画图用的
    std::vector<std::vector<std::pair<double, cv::Point2f>>> axis_list(V.size(),std::vector<std::pair<double, cv::Point2f>>());
    for (size_t vec_idx = 0u; vec_idx < V.size(); vec_idx++) { 
        axis_list[vec_idx] = std::vector<std::pair<double, cv::Point2f>>(candidates.size(), std::make_pair(0, cv::Point2f(0, 0)));
    }
    //画图用的
    std::vector<std::vector<std::pair<double, cv::Point2f>>> uninfomative_axis(V.size(),std::vector<std::pair<double, cv::Point2f>>());
    for (size_t vec_idx = 0u; vec_idx < V.size(); vec_idx++) { 
        uninfomative_axis[vec_idx] = std::vector<std::pair<double, cv::Point2f>>(candidates.size(), std::make_pair(0, cv::Point2f(0, 0)));
    }

    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    //遍历候选点 计算贡献度 加入scores_vec中
    for (size_t i = 0; i < candidates.size(); i++) {

        // Calculate eigenvalues and eigenvectors of the image patch around feature candidate
        int offset = int(area_patch_size/2)+1;
        cv::Rect roi_ij(candidates[i].pc.x - offset, candidates[i].pc.y - offset, area_patch_size+2, area_patch_size+2);
        cv::Mat img_area = img_set.img_intensity(roi_ij);
        cv::Mat eivecim;
        cv::cornerEigenValsAndVecs(img_area, eivecim, 5, 3);
        float eig_val_1 = eivecim.ptr<cv::Vec6f>(offset)[offset][0];
        float eig_val_2 = eivecim.ptr<cv::Vec6f>(offset)[offset][1];

        float i_x, i_y;
        // Eigenvalues come unsorted so we need to figure it out ourselves
        if (eig_val_1 >= eig_val_2) {
            i_x = eivecim.ptr<cv::Vec6f>(offset)[offset][2];
            i_y = eivecim.ptr<cv::Vec6f>(offset)[offset][3];
        } else {
            i_x = eivecim.ptr<cv::Vec6f>(offset)[offset][4];
            i_y = eivecim.ptr<cv::Vec6f>(offset)[offset][5];
        }

        // Approximate gradient of the image patch
        Eigen::Matrix<double, 1, 2> dI_du;
        dI_du << i_x, i_y;


        V3D p = candidates[i].pt;

        MD(2,3) du_dp;
        projectionJacobian(p, du_dp);

        // Calculate score for each direction as shown in formula 4 in paper
        for (size_t vec_idx = 0u; vec_idx < V.size(); vec_idx++) {              
            Eigen::Matrix<double, 2, 1> dpi = du_dp * V[vec_idx];
            dpi.normalize();
            float c_i = fabs(dI_du * dpi);
            scores_vec[vec_idx][i] = std::make_pair(c_i, i);
            axis_list[vec_idx][i] = std::make_pair(c_i, cv::Point2f(i_x, i_y));
            uninfomative_axis[vec_idx][i] = std::make_pair(c_i, cv::Point2f(dpi.x(), dpi.y()));
        }
    }
    //对候选点排序，按照对退化方向贡献度排序
    // Sort scores for each direction
    for (auto& score_vec : scores_vec) {
        std::sort( score_vec.begin(), score_vec.end(),
                        [&]( const std::pair<double, int>& lhs, const std::pair<double, int>& rhs ) {
                        return lhs.first > rhs.first;
                        } );
    }
    //这里是用来画图的，把贡献度大的投影点的二维坐标画出来
    // for (auto& score_vec : axis_list) {
    //     std::sort( score_vec.begin(), score_vec.end(),
    //                     [&]( const std::pair<double, cv::Point2f>& lhs, const std::pair<double, cv::Point2f>& rhs ) {
    //                     return lhs.first > rhs.first;
    //                     } );
    // }
    // 这里是用来画图的，把退化方向画到图像上
    // for (auto& score_vec : uninfomative_axis) {
    //     std::sort( score_vec.begin(), score_vec.end(),
    //                     [&]( const std::pair<double, cv::Point2f>& lhs, const std::pair<double, cv::Point2f>& rhs ) {
    //                     return lhs.first > rhs.first;
    //                     } );
    // }
    cv::Mat moment_col;
    cv::cvtColor(img, moment_col, CV_GRAY2RGB);
    // Select best features for each direction
    std::vector<int> sorted_indices;
    std::vector<float> contributions;
    //排序完后，遍历每个方向，把每个方向的贡献度高的点拿出来，退化时实际就一个方向
    //放入sorted_indices，记录的是索引
    for (size_t idx = 0u; idx < scores_vec[0].size(); ++idx) {
        // Feature candidates that are sorted for each direction with ascending score
        for (size_t vec_idx = 0u; vec_idx < V.size(); vec_idx++) {
            int i = scores_vec[vec_idx][idx].second;
            // Check if feature is already added from another direction
            auto it = std::find(sorted_indices.begin(), sorted_indices.end(), i);
            if (it != sorted_indices.end()) {
                continue;
            }
            sorted_indices.push_back(i);
            
            // Colorize feature candidates based on score
            float score = scores_vec[vec_idx][idx].first;
            contributions.push_back(score);
            //画图的代码
            // float score_x = fabs(axis_list[vec_idx][idx].second.x);
            // float score_y = fabs(axis_list[vec_idx][idx].second.y);
            // std::cout << "score = " << score << ", score_x = " << score_x << ", score_y = " << score_y << std::endl;
            // cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(score*60, 255, 255));
            // cv::Mat hsv_x(1, 1, CV_8UC3, cv::Scalar(score_x*60, 255, 255));
            // cv::Mat hsv_y(1, 1, CV_8UC3, cv::Scalar(score_y*60, 255, 255));
            // cv::Mat bgr;
            // cv::Mat bgr_x;
            // cv::Mat bgr_y;
            // cv::cvtColor(hsv, bgr, CV_HSV2BGR);
            // cv::cvtColor(hsv_x, bgr_x, CV_HSV2BGR);
            // cv::cvtColor(hsv_y, bgr_y, CV_HSV2BGR);
            // cv::Scalar color = cv::Scalar((int)bgr.at<cv::Vec3b>(0, 0)[0],(int)bgr.at<cv::Vec3b>(0, 0)[1],
            //     (int)bgr.at<cv::Vec3b>(0, 0)[2]);
            // cv::Scalar color_x = cv::Scalar((int)bgr_x.at<cv::Vec3b>(0, 0)[0],(int)bgr_x.at<cv::Vec3b>(0, 0)[1],
            //     (int)bgr_x.at<cv::Vec3b>(0, 0)[2]);
            // cv::Scalar color_y = cv::Scalar((int)bgr_y.at<cv::Vec3b>(0, 0)[0],(int)bgr_y.at<cv::Vec3b>(0, 0)[1],
            //     (int)bgr_y.at<cv::Vec3b>(0, 0)[2]);
            // // cv::circle(moment_col, candidates[i].pc, 4, color, 2);
            // cv::arrowedLine(moment_col, candidates[i].pc, cv::Point(20 + candidates[i].pc.x, candidates[i].pc.y), color_x, 2);
            // cv::arrowedLine(moment_col, candidates[i].pc, cv::Point(candidates[i].pc.x, 20 + candidates[i].pc.y), color_y, 2);
            // cv::arrowedLine(moment_col, candidates[i].pc, 
            // cv::Point(candidates[i].pc.x + uninfomative_axis[vec_idx][idx].second.x * 20 ,
            //  candidates[i].pc.y + uninfomative_axis[vec_idx][idx].second.y * 20 ), cv::Scalar(255, 0, 255), 4);
        }
    }
    int add=0;
    //控制加入视觉地图的数量，n_features
    for (size_t idx = 0; idx < sorted_indices.size(); idx++) {

        if (add >+ n_features) {
            break;
        }

        int i = sorted_indices[idx];
        V3D pt = candidates[i].pt;
        V2D pc(new_frame_->w2c(pt));
        int index = static_cast<int>(pc[0]/grid_size)*grid_n_height + static_cast<int>(pc[1]/grid_size);
        // if(grid_cnt[index] >= num_per_grid) continue;

        float* patch = new float[patch_size_total*3];
        getpatch(img, pc, patch, 0);

        getpatch(img, pc, patch, 1);
        getpatch(img, pc, patch, 2);
        PointPtr pt_new(new Point(pt, contributions[idx]));
        Vector3d f = cam->cam2world(pc);
        float cur_value = vk::shiTomasiScore(img, pc[0], pc[1]);
        FeaturePtr ftr_new(new Feature(pc, f, new_frame_->T_f_w_, cur_value, 0));
        ftr_new->img = new_frame_->img_pyr_[0];
        // ftr_new->ImgPyr.resize(5);
        // for(int i=0;i<5;i++) ftr_new->ImgPyr[i] = new_frame_->img_pyr_[i];
        ftr_new->id_ = new_frame_->id_;

        pt_new->addFrameRef(ftr_new);
        pt_new->value = cur_value;
        pt_new->contribution_ = contributions[idx];
        AddPoint(pt_new);
        add += 1;

    }
    memset(grid_cnt, 0, sizeof(int)*length);
    

    if (debug_) {
        //画图加保存图片的代码
        // 创建一个空的 header
        std_msgs::Header empty_header; // 这是一个空的 header，不含任何时间戳或坐标信息
        empty_header.stamp = ros::Time::now();  // 设置当前时间戳
        empty_header.frame_id = "";  // 如果不需要坐标系，保持为空
        sensor_msgs::ImagePtr moment_img =
        cv_bridge::CvImage(empty_header, "bgr8", moment_col).toImageMsg();
        moment_pub.publish(moment_img);
        std::string path = "/home/hzb/temp/feture_" + std::to_string(img_cnt) + ".png";
        std::string raw_path = "/home/hzb/temp/raw_" + std::to_string(img_cnt) + ".png";
        img_cnt++;
        if(candidates.size() >= 15) {
            cv::imwrite(path, moment_col);
            cv::imwrite(raw_path, img);
        }
    }

    printf("[ VIO ]: Add %d 3D points.\n", add);
    // printf("pg.size: %d \n", pg->size());
    // printf("B1. : %.6lf \n", t_b1);
    // printf("B2. : %.6lf \n", t_b2);
    // printf("B3. : %.6lf \n", t_b3);
}
//加入全局地图点
void LidarSelector::AddPoint(PointPtr pt_new)
{
    V3D pt_w(pt_new->pos_[0], pt_new->pos_[1], pt_new->pos_[2]);
    double voxel_size = 0.5;
    float loc_xyz[3];
    for(int j=0; j<3; j++)
    {
      loc_xyz[j] = pt_w[j] / voxel_size;
      if(loc_xyz[j] < 0)
      {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_KEY position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if(iter != feat_map.end())
    {
      iter->second->voxel_points.push_back(pt_new);
      iter->second->count++;
    }
    else
    {
      VOXEL_POINTS *ot = new VOXEL_POINTS(0);
      ot->voxel_points.push_back(pt_new);
      feat_map[position] = ot;
    }
}

void LidarSelector::getWarpMatrixAffine(
    const vk::AbstractCamera& cam,
    const Vector2d& px_ref,
    const Vector3d& f_ref,
    const double depth_ref,
    const SE3& T_cur_ref,
    const int level_ref,    // the corresponding pyrimid level of px_ref
    const int pyramid_level,
    const int halfpatch_size,
    Matrix2d& A_cur_ref)
{
  // Compute affine warp matrix A_ref_cur
  const Vector3d xyz_ref(f_ref*depth_ref);
  Vector3d xyz_du_ref(cam.cam2world(px_ref + Vector2d(halfpatch_size,0)*(1<<level_ref)*(1<<pyramid_level)));
  Vector3d xyz_dv_ref(cam.cam2world(px_ref + Vector2d(0,halfpatch_size)*(1<<level_ref)*(1<<pyramid_level)));
//   Vector3d xyz_du_ref(cam.cam2world(px_ref + Vector2d(halfpatch_size,0)*(1<<level_ref)));
//   Vector3d xyz_dv_ref(cam.cam2world(px_ref + Vector2d(0,halfpatch_size)*(1<<level_ref)));
  xyz_du_ref *= xyz_ref[2]/xyz_du_ref[2];
  xyz_dv_ref *= xyz_ref[2]/xyz_dv_ref[2];
  const Vector2d px_cur(cam.world2cam(T_cur_ref*(xyz_ref)));
  const Vector2d px_du(cam.world2cam(T_cur_ref*(xyz_du_ref)));
  const Vector2d px_dv(cam.world2cam(T_cur_ref*(xyz_dv_ref)));
  A_cur_ref.col(0) = (px_du - px_cur)/halfpatch_size;
  A_cur_ref.col(1) = (px_dv - px_cur)/halfpatch_size;
}

void LidarSelector::warpAffine(
    const Matrix2d& A_cur_ref,
    const cv::Mat& img_ref,
    const Vector2d& px_ref,
    const int level_ref,
    const int search_level,
    const int pyramid_level,
    const int halfpatch_size,
    float* patch)
{
  const int patch_size = halfpatch_size*2 ;
  const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
  if(isnan(A_ref_cur(0,0)))
  {
    printf("Affine warp is NaN, probably camera has no translation\n"); // TODO
    return;
  }
//   Perform the warp on a larger patch.
//   float* patch_ptr = patch;
//   const Vector2f px_ref_pyr = px_ref.cast<float>() / (1<<level_ref) / (1<<pyramid_level);
//   const Vector2f px_ref_pyr = px_ref.cast<float>() / (1<<level_ref);
  for (int y=0; y<patch_size; ++y)
  {
    for (int x=0; x<patch_size; ++x)//, ++patch_ptr)
    {
      // P[patch_size_total*level + x*patch_size+y]
      Vector2f px_patch(x-halfpatch_size, y-halfpatch_size);
      px_patch *= (1<<search_level);
      px_patch *= (1<<pyramid_level);
      const Vector2f px(A_ref_cur*px_patch + px_ref.cast<float>());
      if (px[0]<0 || px[1]<0 || px[0]>=img_ref.cols-1 || px[1]>=img_ref.rows-1)
        patch[patch_size_total*pyramid_level + y*patch_size+x] = 0;
        // *patch_ptr = 0;
      else
        patch[patch_size_total*pyramid_level + y*patch_size+x] = (float) vk::interpolateMat_8u(img_ref, px[0], px[1]);
        // *patch_ptr = (uint8_t) vk::interpolateMat_8u(img_ref, px[0], px[1]);
    }
  }
}

double LidarSelector::NCC(float* ref_patch, float* cur_patch, int patch_size)
{    
    double sum_ref = std::accumulate(ref_patch, ref_patch + patch_size, 0.0);
    double mean_ref =  sum_ref / patch_size;

    double sum_cur = std::accumulate(cur_patch, cur_patch + patch_size, 0.0);
    double mean_curr =  sum_cur / patch_size;

    double numerator = 0, demoniator1 = 0, demoniator2 = 0;
    for (int i = 0; i < patch_size; i++) 
    {
        double n = (ref_patch[i] - mean_ref) * (cur_patch[i] - mean_curr);
        numerator += n;
        demoniator1 += (ref_patch[i] - mean_ref) * (ref_patch[i] - mean_ref);
        demoniator2 += (cur_patch[i] - mean_curr) * (cur_patch[i] - mean_curr);
    }
    return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);
}

int LidarSelector::getBestSearchLevel(
    const Matrix2d& A_cur_ref,
    const int max_level)
{
  // Compute patch level in other image
  int search_level = 0;
  double D = A_cur_ref.determinant();

  while(D > 3.0 && search_level < max_level)
  {
    search_level += 1;
    D *= 0.25;
  }
  return search_level;
}

#ifdef FeatureAlign
void LidarSelector::createPatchFromPatchWithBorder(float* patch_with_border, float* patch_ref)
{
  float* ref_patch_ptr = patch_ref;
  for(int y=1; y<patch_size+1; ++y, ref_patch_ptr += patch_size)
  {
    float* ref_patch_border_ptr = patch_with_border + y*(patch_size+2) + 1;
    for(int x=0; x<patch_size; ++x)
      ref_patch_ptr[x] = ref_patch_border_ptr[x];
  }
}
#endif
//从视觉全局地图中选取视觉子图，与图像构建光度残差
//入参是当前图像和当前lidar
void LidarSelector::addFromSparseMap(cv::Mat img, PointCloudXYZI::Ptr pg)
{
    if(feat_map.size()<=0) return;
    // double ts0 = omp_get_wtime();

    pg_down->reserve(feat_map.size());
    downSizeFilter.setInputCloud(pg);
    downSizeFilter.filter(*pg_down);
    
    reset_grid();
    memset(map_value, 0, sizeof(float)*length);

    sub_sparse_map->reset();
    deque< PointPtr >().swap(sub_map_cur_frame_);

    float voxel_size = 0.5;
    
    unordered_map<VOXEL_KEY, float>().swap(sub_feat_map);
    unordered_map<int, Warp*>().swap(Warp_map);

    cv::Mat depth_img = cv::Mat::zeros(height, width, CV_32FC1);
    float* it = (float*)depth_img.data;

    double t_insert, t_depth, t_position;
    t_insert=t_depth=t_position=0;

    int loc_xyz[3];
    vector<PointPtr> new_voxel_points;
    new_voxel_points.reserve(length);
    // printf("A0. initial depthmap: %.6lf \n", omp_get_wtime() - ts0);
    // double ts1 = omp_get_wtime();
    //这一步就是遍历lidar，看lidar大概落在全局视觉地图的那些体素上，方便快速从视觉全局地图中选取
    //相机FOV内的视觉子图，主要就是标记一下sub_feat_map
    for(int i=0; i<pg_down->size(); i++)
    {
        // Transform Point to world coordinate
        V3D pt_w(pg_down->points[i].x, pg_down->points[i].y, pg_down->points[i].z);

        // Determine the key of hash table      
        for(int j=0; j<3; j++)
        {
            loc_xyz[j] = floor(pt_w[j] / voxel_size);
        }
        VOXEL_KEY position(loc_xyz[0], loc_xyz[1], loc_xyz[2]);
        auto corre_voxel = feat_map.find(position);
        // if(corre_voxel != feat_map.end()) {
        //     std::vector<PointPtr> &voxel_points = corre_voxel->second->voxel_points;
        //     if(voxel_points.size() == 0) {
        //         feat_map.erase(corre_voxel);  // 直接删除当前元素
        //         // ROS_ERROR("----remove voxel-----");
        //     }
        // }

        auto iter = sub_feat_map.find(position);
        if(iter == sub_feat_map.end())
        {
            sub_feat_map[position] = 1.0;
        }
                    
        V3D pt_c(new_frame_->w2f(pt_w));

        V2D px;
        if(pt_c[2] > 0)
        {
            px[0] = fx * pt_c[0]/pt_c[2] + cx;
            px[1] = fy * pt_c[1]/pt_c[2] + cy;

            if(new_frame_->cam_->isInFrame(px.cast<int>(), (patch_size_half+1)*8))
            {
                float depth = pt_c[2];
                int col = int(px[0]);
                int row = int(px[1]);
                it[width*row+col] = depth;        
            }
        }
    }
    

    // double t1 = omp_get_wtime();
    //sub_feat_map就是lidar目前大概覆盖的视觉全局地图的体素
    //然后遍历sub_feat_map，从视觉全局地图中找到对应体素
    //一个体素中有很多点，这里先都加入到new_voxel_points，注意有FOV check
    for(auto& iter : sub_feat_map)
    {   
        VOXEL_KEY position = iter.first;
        // double t4 = omp_get_wtime();
        auto corre_voxel = feat_map.find(position);
        // double t5 = omp_get_wtime();

        if(corre_voxel != feat_map.end())
        {
            std::vector<PointPtr> &voxel_points = corre_voxel->second->voxel_points;
            int voxel_num = voxel_points.size();
            float cout_dist = 0;
            //遍历一个体素中所有点
            for (int i=0; i<voxel_num; i++)
            {
                PointPtr pt = voxel_points[i];
                if(pt==nullptr) continue;
                V3D pt_cam(new_frame_->w2f(pt->pos_));
                if(pt_cam[2]<0) continue;

                V2D pc(new_frame_->w2c(pt->pos_));

                FeaturePtr ref_ftr;
                //如果在FOV内，则加入到new_voxel_points
                //其余代码可以忽略，因为原FAST-LIVO分网格，每个网格只要一个，而且还
                //计算深度，把深度突变的点去点，洞里就不用了
                if(new_frame_->cam_->isInFrame(pc.cast<int>(), (patch_size_half+1)*8)) // 20px is the patch size in the matcher
                {
                    //这一行是关键
                    new_voxel_points.push_back(pt);
                    //后面的可以不看，网格的用法我没用
                    int index = static_cast<int>(pc[0]/grid_size)*grid_n_height + static_cast<int>(pc[1]/grid_size);
                    grid_num[index] = TYPE_MAP;
                    Vector3d obs_vec(new_frame_->pos() - pt->pos_);

                    float cur_dist = obs_vec.norm();
                    if(cur_dist > cout_dist) {
                        cout_dist = cur_dist;
                    }
                    float cur_value = pt->value;

                    if (cur_dist <= map_dist[index]) 
                    {
                        map_dist[index] = cur_dist;
                        voxel_points_[index] = pt;
                        // contri_values[index] = pt->contribution_;
                    } 

                    if (cur_value >= map_value[index])
                    {
                        map_value[index] = cur_value;
                    }
                } 
            }  
        } 
    }
        
    // double t2 = omp_get_wtime();

    // cout<<"B. feat_map.find: "<<t2-t1<<endl;

    double t_2, t_3, t_4, t_5;
    t_2=t_3=t_4=t_5=0;

    int add_to_submap = 0;
    cv::Mat submap_high;
    cv::Mat submap_all;
    cv::cvtColor(img, submap_high, CV_GRAY2RGB);
    cv::cvtColor(img, submap_all, CV_GRAY2RGB);
    //这里就对FOV内使用的视觉地图点，按照贡献度排序了
    std::sort( new_voxel_points.begin(), new_voxel_points.end(),
        [&]( const PointPtr& lhs, const PointPtr& rhs )
        {
            return lhs->contribution_ > rhs->contribution_;
        } );
    //画图用的
    // for(int i = 0; i < new_voxel_points.size(); i++) {
    //     PointPtr pt = new_voxel_points[i];
    //     if(pt==nullptr) {
    //         ROS_ERROR("pt is null, and i = %d", i);
    //         continue;
    //     }
    //     float score = pt->contribution_;
    //     V2D pc(new_frame_->w2c(pt->pos_));
    //     cv::Point img_pc(pc[0], pc[1]);
    //     cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(score*60, 255, 255));
    //     cv::Mat bgr;
    //     cv::cvtColor(hsv, bgr, CV_HSV2BGR);
    //     cv::Scalar color = cv::Scalar((int)bgr.at<cv::Vec3b>(0, 0)[0],(int)bgr.at<cv::Vec3b>(0, 0)[1],
    //                     (int)bgr.at<cv::Vec3b>(0, 0)[2]);
    //     cv::circle(submap_all, img_pc, 4, color, 2);
    // }
    // for (int i=0; i<length; i++) 
    // std::cout << "new_voxel_points size =" << new_voxel_points.size() << std::endl;
    
    //排序结束后，加入视觉子图，用add_submap_num_控制加入的点的数量
    for(int i = 0; i < new_voxel_points.size(); i++)
    { 
        
            if(add_to_submap > add_submap_num_) break;
            PointPtr pt = new_voxel_points[i];

            if(pt==nullptr) {
                ROS_ERROR("pt is null, and i = %d", i);
                continue;}
            // ROS_ERROR("---------pt->contribution_ = %f", pt->contribution_);

            V2D pc(new_frame_->w2c(pt->pos_));
            V3D pt_cam(new_frame_->w2f(pt->pos_));
   
            bool depth_continous = false;
            
            FeaturePtr ref_ftr;

            if(!pt->getCloseViewObs(new_frame_->pos(), ref_ftr, pc)) continue;

            // t_3 += omp_get_wtime() - t_1;

            std::vector<float> patch_wrap(patch_size_total * 3);

            // patch_wrap = ref_ftr->patch;

            // t_1 = omp_get_wtime();
           
            int search_level;
            Matrix2d A_cur_ref_zero;

            auto iter_warp = Warp_map.find(ref_ftr->id_);
            if(iter_warp != Warp_map.end())
            {
                search_level = iter_warp->second->search_level;
                A_cur_ref_zero = iter_warp->second->A_cur_ref;
            }
            else
            {
                getWarpMatrixAffine(*cam, ref_ftr->px, ref_ftr->f, (ref_ftr->pos() - pt->pos_).norm(), 
                new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse(), 0, 0, patch_size_half, A_cur_ref_zero);
                
                search_level = getBestSearchLevel(A_cur_ref_zero, 2);

                Warp *ot = new Warp(search_level, A_cur_ref_zero);
                Warp_map[ref_ftr->id_] = ot;
            }

            // t_4 += omp_get_wtime() - t_1;

            // t_1 = omp_get_wtime();

            for(int pyramid_level=0; pyramid_level<=2; pyramid_level++)
            {                
                warpAffine(A_cur_ref_zero, ref_ftr->img, ref_ftr->px, ref_ftr->level, search_level, pyramid_level, patch_size_half, patch_wrap.data());
            }

            getpatch(img, pc, patch_cache.data(), 0);

            if(ncc_en)
            {
                double ncc = NCC(patch_wrap.data(), patch_cache.data(), patch_size_total);
                if(ncc < ncc_thre) continue;
            }

            float error = 0.0;
            for (int ind=0; ind<patch_size_total; ind++) 
            {
                error += (patch_wrap[ind]-patch_cache[ind]) * (patch_wrap[ind]-patch_cache[ind]);
            }
            if(error > outlier_threshold*patch_size_total) continue;
            add_to_submap++;
            float score = pt->contribution_;

            sub_map_cur_frame_.push_back(pt);
            cv::Point img_pc(pc[0], pc[1]);
            cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(score*60, 255, 255));
            cv::Mat bgr;
            cv::cvtColor(hsv, bgr, CV_HSV2BGR);
            cv::Scalar color = cv::Scalar((int)bgr.at<cv::Vec3b>(0, 0)[0],(int)bgr.at<cv::Vec3b>(0, 0)[1],
                            (int)bgr.at<cv::Vec3b>(0, 0)[2]);
            cv::circle(submap_high, img_pc, 4, color, 2);
            sub_sparse_map->propa_errors.push_back(error);
            sub_sparse_map->search_levels.push_back(search_level);
            sub_sparse_map->errors.push_back(error);
            sub_sparse_map->index.push_back(i);  
            sub_sparse_map->voxel_points.push_back(pt);
            sub_sparse_map->patch.push_back(std::move(patch_wrap));
            // t_5 += omp_get_wtime() - t_1;
        // }
    }
    //画图用的
    if(show_submap_) {
        std::string all_path = "/home/hzb/temp/a_all" + std::to_string(save_cnt) + ".jpg";
        std::string high_path = "/home/hzb/temp/a_high" + std::to_string(save_cnt) + ".jpg";
        save_cnt++;
            cv::imwrite(all_path, submap_all);
            cv::imwrite(high_path, submap_high);    
    }
    // double t3 = omp_get_wtime();
    // cout<<"C. addSubSparseMap: "<<t3-t2<<endl;
    // cout<<"depthcontinuous: C1 "<<t_2<<" C2 "<<t_3<<" C3 "<<t_4<<" C4 "<<t_5<<endl;
    printf("[ VIO ]: choose %d points from sub_sparse_map.\n", int(sub_sparse_map->index.size()));
}

#ifdef FeatureAlign
bool LidarSelector::align2D(
    const cv::Mat& cur_img,
    float* ref_patch_with_border,
    float* ref_patch,
    const int n_iter,
    Vector2d& cur_px_estimate,
    int index)
{
#ifdef __ARM_NEON__
  if(!no_simd)
    return align2D_NEON(cur_img, ref_patch_with_border, ref_patch, n_iter, cur_px_estimate);
#endif

  const int halfpatch_size_ = 4;
  const int patch_size_ = 8;
  const int patch_area_ = 64;
  bool converged=false;

  // compute derivative of template and prepare inverse compositional
  float __attribute__((__aligned__(16))) ref_patch_dx[patch_area_];
  float __attribute__((__aligned__(16))) ref_patch_dy[patch_area_];
  Matrix3f H; H.setZero();

  // compute gradient and hessian
  const int ref_step = patch_size_+2;
  float* it_dx = ref_patch_dx;
  float* it_dy = ref_patch_dy;
  for(int y=0; y<patch_size_; ++y) 
  {
    float* it = ref_patch_with_border + (y+1)*ref_step + 1; 
    for(int x=0; x<patch_size_; ++x, ++it, ++it_dx, ++it_dy)
    {
      Vector3f J;
      J[0] = 0.5 * (it[1] - it[-1]); 
      J[1] = 0.5 * (it[ref_step] - it[-ref_step]); 
      J[2] = 1; 
      *it_dx = J[0];
      *it_dy = J[1];
      H += J*J.transpose(); 
    }
  }
  Matrix3f Hinv = H.inverse();
  float mean_diff = 0;

  // Compute pixel location in new image:
  float u = cur_px_estimate.x();
  float v = cur_px_estimate.y();

  // termination condition
  const float min_update_squared = 0.03*0.03;//0.03*0.03
  const int cur_step = cur_img.step.p[0];
  float chi2 = 0;
  chi2 = sub_sparse_map->propa_errors[index];
  Vector3f update; update.setZero();
  for(int iter = 0; iter<n_iter; ++iter)
  {
    int u_r = floor(u);
    int v_r = floor(v);
    if(u_r < halfpatch_size_ || v_r < halfpatch_size_ || u_r >= cur_img.cols-halfpatch_size_ || v_r >= cur_img.rows-halfpatch_size_)
      break;

    if(isnan(u) || isnan(v)) // TODO very rarely this can happen, maybe H is singular? should not be at corner.. check
      return false;

    // compute interpolation weights
    float subpix_x = u-u_r;
    float subpix_y = v-v_r;
    float wTL = (1.0-subpix_x)*(1.0-subpix_y);
    float wTR = subpix_x * (1.0-subpix_y);
    float wBL = (1.0-subpix_x)*subpix_y;
    float wBR = subpix_x * subpix_y;

    // loop through search_patch, interpolate
    float* it_ref = ref_patch;
    float* it_ref_dx = ref_patch_dx;
    float* it_ref_dy = ref_patch_dy;
    float new_chi2 = 0.0;
    Vector3f Jres; Jres.setZero();
    for(int y=0; y<patch_size_; ++y)
    {
      uint8_t* it = (uint8_t*) cur_img.data + (v_r+y-halfpatch_size_)*cur_step + u_r-halfpatch_size_; 
      for(int x=0; x<patch_size_; ++x, ++it, ++it_ref, ++it_ref_dx, ++it_ref_dy)
      {
        float search_pixel = wTL*it[0] + wTR*it[1] + wBL*it[cur_step] + wBR*it[cur_step+1];
        float res = search_pixel - *it_ref + mean_diff;
        Jres[0] -= res*(*it_ref_dx);
        Jres[1] -= res*(*it_ref_dy);
        Jres[2] -= res;
        new_chi2 += res*res;
      }
    }

    if(iter > 0 && new_chi2 > chi2)
    {
    //   cout << "error increased." << endl;
      u -= update[0];
      v -= update[1];
      break;
    }
    chi2 = new_chi2;

    sub_sparse_map->align_errors[index] = new_chi2;

    update = Hinv * Jres;
    u += update[0];
    v += update[1];
    mean_diff += update[2];

#if SUBPIX_VERBOSE
    cout << "Iter " << iter << ":"
         << "\t u=" << u << ", v=" << v
         << "\t update = " << update[0] << ", " << update[1]
//         << "\t new chi2 = " << new_chi2 << endl;
#endif

    if(update[0]*update[0]+update[1]*update[1] < min_update_squared)
    {
#if SUBPIX_VERBOSE
      cout << "converged." << endl;
#endif
      converged=true;
      break;
    }
  }

  cur_px_estimate << u, v;
  return converged;
}

void LidarSelector::FeatureAlignment(cv::Mat img)
{
    int total_points = sub_sparse_map->index.size();
    if (total_points==0) return;
    memset(align_flag, 0, length);
    int FeatureAlignmentNum = 0;
       
    for (int i=0; i<total_points; i++) 
    {
        bool res;
        int search_level = sub_sparse_map->search_levels[i];
        Vector2d px_scaled(sub_sparse_map->px_cur[i]/(1<<search_level));
        res = align2D(new_frame_->img_pyr_[search_level], sub_sparse_map->patch_with_border[i], sub_sparse_map->patch[i],
                        20, px_scaled, i);
        sub_sparse_map->px_cur[i] = px_scaled * (1<<search_level);
        if(res)
        {
            align_flag[i] = 1;
            FeatureAlignmentNum++;
        }
    }
}
#endif

//视觉ESIKF更新
float LidarSelector::UpdateState(cv::Mat img, float total_residual, int level, const std::vector<V3D>& V) 
{
    int total_points = sub_sparse_map->index.size();
    if (total_points==0) return 0.;
    StatesGroup old_state = (*state);
    V2D pc; 
    MD(1,2) Jimg;
    MD(2,3) Jdpi;
    MD(1,3) Jdphi, Jdp, JdR, Jdt;
    VectorXd z;
    // VectorXd R;
    bool EKF_end = false;
    /* Compute J */
    float error=0.0, last_error=total_residual, patch_error=0.0, last_patch_error=0.0, propa_error=0.0;
    // MatrixXd H;
    bool z_init = true;
    const int H_DIM = total_points * patch_size_total;
    
    // K.resize(H_DIM, H_DIM);
    z.resize(H_DIM);
    z.setZero();
    // R.resize(H_DIM);
    // R.setZero();

    // H.resize(H_DIM, DIM_STATE);
    // H.setZero();
    H_sub.resize(H_DIM, 6);
    H_sub.setZero();
    //ESIKF迭代次数
    for (int iteration=0; iteration<NUM_MAX_ITERATIONS; iteration++) 
    {
        // double t1 = omp_get_wtime();
        double count_outlier = 0;
     
        error = 0.0;
        propa_error = 0.0;
        n_meas_ =0;
        M3D Rwi(state->rot_end);
        V3D Pwi(state->pos_end);
        Rcw = Rci * Rwi.transpose();
        Pcw = -Rci*Rwi.transpose()*Pwi + Pci;
        Jdp_dt = Rci * Rwi.transpose();
        
        M3D p_hat;
        int i;
        //遍历视觉地图中的点，计算光度误差
        for (i=0; i<sub_sparse_map->index.size(); i++) 
        {
            patch_error = 0.0;
            int search_level = sub_sparse_map->search_levels[i];
            int pyramid_level = level + search_level;
            const int scale =  (1<<pyramid_level);
            
            PointPtr pt = sub_sparse_map->voxel_points[i];

            if(pt==nullptr) continue;

            V3D pf = Rcw * pt->pos_ + Pcw;
            pc = cam->world2cam(pf);
            // if((level==2 && iteration==0) || (level==1 && iteration==0) || level==0)
            {
                dpi(pf, Jdpi);
                p_hat << SKEW_SYM_MATRX(pf);
            }
            const float u_ref = pc[0];
            const float v_ref = pc[1];
            const int u_ref_i = floorf(pc[0]/scale)*scale; 
            const int v_ref_i = floorf(pc[1]/scale)*scale;
            const float subpix_u_ref = (u_ref-u_ref_i)/scale;
            const float subpix_v_ref = (v_ref-v_ref_i)/scale;
            const float w_ref_tl = (1.0-subpix_u_ref) * (1.0-subpix_v_ref);
            const float w_ref_tr = subpix_u_ref * (1.0-subpix_v_ref);
            const float w_ref_bl = (1.0-subpix_u_ref) * subpix_v_ref;
            const float w_ref_br = subpix_u_ref * subpix_v_ref;
            
            vector<float> P = sub_sparse_map->patch[i];
            // V3D weak_dir;
            // if(V.size() == 1) {
            //     weak_dir= V[0];
            // }
            for (int x=0; x<patch_size; x++) 
            {
                uint8_t* img_ptr = (uint8_t*) img.data + (v_ref_i+x*scale-patch_size_half*scale)*width + u_ref_i-patch_size_half*scale;
                for (int y=0; y<patch_size; ++y, img_ptr+=scale) 
                {
                    // if((level==2 && iteration==0) || (level==1 && iteration==0) || level==0)
                    //{
                    float du = 0.5f * ((w_ref_tl*img_ptr[scale] + w_ref_tr*img_ptr[scale*2] + w_ref_bl*img_ptr[scale*width+scale] + w_ref_br*img_ptr[scale*width+scale*2])
                                -(w_ref_tl*img_ptr[-scale] + w_ref_tr*img_ptr[0] + w_ref_bl*img_ptr[scale*width-scale] + w_ref_br*img_ptr[scale*width]));
                    float dv = 0.5f * ((w_ref_tl*img_ptr[scale*width] + w_ref_tr*img_ptr[scale+scale*width] + w_ref_bl*img_ptr[width*scale*2] + w_ref_br*img_ptr[width*scale*2+scale])
                                -(w_ref_tl*img_ptr[-scale*width] + w_ref_tr*img_ptr[-scale*width+scale] + w_ref_bl*img_ptr[0] + w_ref_br*img_ptr[scale]));
                    Jimg << du, dv;
                    Jimg = Jimg * (1.0/scale);
                    Jdphi = Jimg * Jdpi * p_hat;
                    Jdp = -Jimg * Jdpi;
                    JdR = Jdphi * Jdphi_dR + Jdp * Jdp_dR;
                    Jdt = Jdp * Jdp_dt;
                    // if(V.size() == 1) {
                    //     double sum = Jdt.norm();
                    //     Jdt[0] = Jdt[0] > 0 ? sum*abs(weak_dir[0]) : -sum*abs(weak_dir[0]);
                    //     Jdt[1] = Jdt[1] > 0 ? sum*abs(weak_dir[1]) : -sum*abs(weak_dir[1]);
                    //     Jdt[2] = Jdt[2] > 0 ? sum*abs(weak_dir[2]) : -sum*abs(weak_dir[2]);
                    //     // std::cout << "jdt = " << Jdt << std::endl;
                    // }
                    //}
                    double res = w_ref_tl*img_ptr[0] + w_ref_tr*img_ptr[scale] + w_ref_bl*img_ptr[scale*width] + w_ref_br*img_ptr[scale*width+scale]  - P[patch_size_total*level + x*patch_size+y];
                    z(i*patch_size_total+x*patch_size+y) = res;
                    // float weight = 1.0;
                    // if(iteration > 0)
                    //     weight = weight_function_->value(res/weight_scale_); 
                    // R(i*patch_size_total+x*patch_size+y) = weight;       
                    patch_error +=  res*res;
                    n_meas_++;
                    // H.block<1,6>(i*patch_size_total+x*patch_size+y,0) << JdR*weight, Jdt*weight;
                    // if((level==2 && iteration==0) || (level==1 && iteration==0) || level==0)
                    H_sub.block<1,6>(i*patch_size_total+x*patch_size+y,0) << JdR * img_scale_, Jdt * img_scale_;
                }
            }  

            sub_sparse_map->errors[i] = patch_error * img_scale_;
            error += patch_error * img_scale_;
        }

        // computeH += omp_get_wtime() - t1;

        error = error/n_meas_;

        // double t3 = omp_get_wtime();

        if (error <= last_error) 
        {
            old_state = (*state);
            last_error = error;

            // K = (H.transpose() / img_point_cov * H + state->cov.inverse()).inverse() * H.transpose() / img_point_cov;
            // auto vec = (*state_propagat) - (*state);
            // G = K*H;
            // (*state) += (-K*z + vec - G*vec);

            auto &&H_sub_T = H_sub.transpose();
            H_T_H.block<6,6>(0,0) = H_sub_T * H_sub;
            MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
            auto &&HTz = H_sub_T * z;
            // K = K_1.block<DIM_STATE,6>(0,0) * H_sub_T;
            auto vec = (*state_propagat) - (*state);
            G.block<DIM_STATE,6>(0,0) = K_1.block<DIM_STATE,6>(0,0) * H_T_H.block<6,6>(0,0);
            VD(DIM_STATE) solution = - K_1.block<DIM_STATE,6>(0,0) * HTz + vec - G.block<DIM_STATE,6>(0,0) * vec.block<6,1>(0,0);
            (*state) += solution;
            auto &&rot_add = solution.block<3,1>(0,0);
            auto &&t_add   = solution.block<3,1>(3,0);

            if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f))
            {
                EKF_end = true;
            }
        }
        else
        {
            (*state) = old_state;
            EKF_end = true;
        }

        // ekf_time += omp_get_wtime() - t3;

        if (iteration==NUM_MAX_ITERATIONS || EKF_end) 
        {
            break;
        }
    }
    return last_error;
} 

void LidarSelector::updateFrameState(StatesGroup state)
{
    M3D Rwi(state.rot_end);
    V3D Pwi(state.pos_end);
    Rcw = Rci * Rwi.transpose();
    Pcw = -Rci*Rwi.transpose()*Pwi + Pci;
    new_frame_->T_f_w_ = SE3(Rcw, Pcw);
}

void LidarSelector::addObservation(cv::Mat img)
{
    int total_points = sub_sparse_map->index.size();
    if (total_points==0) return;

    for (int i=0; i<total_points; i++) 
    {
        PointPtr pt = sub_sparse_map->voxel_points[i];
        if(pt==nullptr) continue;
        V2D pc(new_frame_->w2c(pt->pos_));
        SE3 pose_cur = new_frame_->T_f_w_;
        bool add_flag = false;
        // if (sub_sparse_map->errors[i]<= 100*patch_size_total && sub_sparse_map->errors[i]>0)
        {
            //TODO: condition: distance and view_angle 
            // Step 1: time
            FeaturePtr last_feature =  pt->obs_.back();
            // if(new_frame_->id_ >= last_feature->id_ + 20) add_flag = true;

            // Step 2: delta_pose
            SE3 pose_ref = last_feature->T_f_w_;
            SE3 delta_pose = pose_ref * pose_cur.inverse();
            double delta_p = delta_pose.translation().norm();
            double delta_theta = (delta_pose.rotation_matrix().trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (delta_pose.rotation_matrix().trace() - 1));            
            if(delta_p > 0.5 || delta_theta > 10) add_flag = true;

            // Step 3: pixel distance
            Vector2d last_px = last_feature->px;
            double pixel_dist = (pc-last_px).norm();
            if(pixel_dist > 40) add_flag = true;
            
            // Maintain the size of 3D Point observation features.
            if(pt->obs_.size()>=20)
            {
                FeaturePtr ref_ftr;
                pt->getFurthestViewObs(new_frame_->pos(), ref_ftr);
                pt->deleteFeatureRef(ref_ftr);
                // ROS_WARN("ref_ftr->id_ is %d", ref_ftr->id_);
            } 
            if(add_flag)
            {
                pt->value = vk::shiTomasiScore(img, pc[0], pc[1]);
                Vector3d f = cam->cam2world(pc);
                FeaturePtr ftr_new(new Feature(pc, f, new_frame_->T_f_w_, pt->value, sub_sparse_map->search_levels[i])); 
                ftr_new->img = new_frame_->img_pyr_[0];
                ftr_new->id_ = new_frame_->id_;
                // ftr_new->ImgPyr.resize(5);
                // for(int i=0;i<5;i++) ftr_new->ImgPyr[i] = new_frame_->img_pyr_[i];
                pt->addFrameRef(ftr_new);      
            }
        }
    }
}
//ESIKF 迭代计算视觉子图与图像的光度残差
void LidarSelector::ComputeJ(cv::Mat img, const std::vector<V3D>& V) 
{
    int total_points = sub_sparse_map->index.size();
    if (total_points==0) return;
    float error = 1e10;
    float now_error = error;
    StatesGroup temp_state = (*state);
    for (int level=2; level>=0; level--) 
    {
        now_error = UpdateState(img, error, level, V);
    }
    if (now_error < error)
    {
        state->cov -= G*state->cov;
    }
    V3D pos_err((*state).pos_end - temp_state.pos_end);
    V3D weak_g;
    if(V.size() == 1) {
        weak_g = V[0];
        weak_g[0] = -weak_g[0];
        weak_g[1] = -weak_g[1];
        weak_g[2] = -weak_g[2];
    } else {
        weak_g = V3D(0,0,0);
    }
    // std::cout << "------- weak_g: " << weak_g.transpose() << std::endl; 
    V3D prj = abs(pos_err.dot(weak_g)) * weak_g;
    if(V.size() ==1 && weak_g.dot(pos_err) < 0) {
        // std::cout << "-----------------weak_g: " << weak_g.transpose() <<  std::endl;
        // std::cout << "-----------------pos_err: " << pos_err.transpose() << std::endl;
        // std::cout << "-----------------prj: " << prj.transpose() << std::endl;
        // std::cout << "-----------------pos_err: " << pos_err.transpose() << std::endl;
        // V3D temp_pos(pos_err);
        // temp_pos[1] = -temp_pos[1];
        (*state).pos_end = temp_state.pos_end + prj;
    }
    //  else if(pos_err.norm() < 0.005) {
    //     (*state).pos_end = temp_state.pos_end + 0.005*weak_g;
    // }
    geometry_msgs::PoseStamped calib_pose;
    calib_pose.header.stamp = ros::Time::now();
    calib_pose.header.frame_id = "camera_odom_frame";
    calib_pose.pose.position.x = pos_err.norm();
    calib_pose.pose.position.y = prj.norm();
    pose_pub.publish(calib_pose);
    
    updateFrameState(*state);
}

void LidarSelector::display_keypatch(double time)
{
    int total_points = sub_sparse_map->index.size();
    if (total_points==0) return;
    for(int i=0; i<total_points; i++)
    {
        PointPtr pt = sub_sparse_map->voxel_points[i];
        V2D pc(new_frame_->w2c(pt->pos_));
        cv::Point2f pf;
        pf = cv::Point2f(pc[0], pc[1]); 
        if (sub_sparse_map->errors[i]<8000) // 5.5
            cv::circle(img_cp, pf, 6, cv::Scalar(0, 255, 0), -1, 8); // Green Sparse Align tracked
        else
            cv::circle(img_cp, pf, 6, cv::Scalar(255, 0, 0), -1, 8); // Blue Sparse Align tracked
    }   
    std::string text = std::to_string(int(1/time))+" HZ";
    cv::Point2f origin;
    origin.x = 20;
    origin.y = 20;
    cv::putText(img_cp, text, origin, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, 8, 0);
}

V3F LidarSelector::getpixel(cv::Mat img, V2D pc) 
{
    const float u_ref = pc[0];
    const float v_ref = pc[1];
    const int u_ref_i = floorf(pc[0]); 
    const int v_ref_i = floorf(pc[1]);
    const float subpix_u_ref = (u_ref-u_ref_i);
    const float subpix_v_ref = (v_ref-v_ref_i);
    const float w_ref_tl = (1.0-subpix_u_ref) * (1.0-subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0-subpix_v_ref);
    const float w_ref_bl = (1.0-subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;
    uint8_t* img_ptr = (uint8_t*) img.data + ((v_ref_i)*width + (u_ref_i))*3;
    float B = w_ref_tl*img_ptr[0] + w_ref_tr*img_ptr[0+3] + w_ref_bl*img_ptr[width*3] + w_ref_br*img_ptr[width*3+0+3];
    float G = w_ref_tl*img_ptr[1] + w_ref_tr*img_ptr[1+3] + w_ref_bl*img_ptr[1+width*3] + w_ref_br*img_ptr[width*3+1+3];
    float R = w_ref_tl*img_ptr[2] + w_ref_tr*img_ptr[2+3] + w_ref_bl*img_ptr[2+width*3] + w_ref_br*img_ptr[width*3+2+3];
    V3F pixel(B,G,R);
    return pixel;
}

//核心函数，主要四部
// 1 addFromSparseMap 选取视觉子图，这个主要用于与当前图像计算光度残差 根据贡献度选
// 2  addSparseMap 从当前帧图像选取点加入视觉全局地图，按照贡献度
//3 ComputeJ ESIKF 计算视觉子图与当前图像的光度残差
// 4addObservation 更新地图点的当前观测，这个没改
void LidarSelector::detect(cv::Mat img, PointCloudXYZI::Ptr pg, const std::vector<V3D>& V, const std::vector<V3D>& V_world) 
{
    if(width!=img.cols || height!=img.rows)
    {
        // std::cout<<"Resize the img scale !!!"<<std::endl;
        double scale = 0.5;
        cv::resize(img,img,cv::Size(img.cols*scale,img.rows*scale),0,0,CV_INTER_LINEAR);
    }
    img_rgb = img.clone();
    img_cp = img.clone();
    cv::cvtColor(img,img,CV_BGR2GRAY);

    new_frame_.reset(new Frame(cam, img.clone()));
    updateFrameState(*state);

    if(stage_ == STAGE_FIRST_FRAME && pg->size()>10)
    {
        new_frame_->setKeyframe();
        stage_ = STAGE_DEFAULT_FRAME;
    }

    double t1 = omp_get_wtime();

    addFromSparseMap(img, pg);

    double t3 = omp_get_wtime();

    addSparseMap(img, pg, V);

    double t4 = omp_get_wtime();
    
    // computeH = ekf_time = 0.0;
    
    ComputeJ(img, V_world);

    double t5 = omp_get_wtime();

    addObservation(img);
    
    double t2 = omp_get_wtime();
    
    frame_count ++;
    ave_total = ave_total * (frame_count - 1) / frame_count + (t2 - t1) / frame_count;

    printf("[ VIO ]: time: addFromSparseMap: %.6f addSparseMap: %.6f ComputeJ: %.6f addObservation: %.6f total time: %.6f ave_total: %.6f.\n"
    , t3-t1, t4-t3, t5-t4, t2-t5, t2-t1, ave_total);

    display_keypatch(t2-t1);
} 

} // namespace lidar_selection