//
// Created by lab on 2021/12/8.
//

#ifndef CHECKERBOARD_LC_CALIB_PINHOLECAM_H
#define CHECKERBOARD_LC_CALIB_PINHOLECAM_H


class PinholeCam
{
public:
    PinholeCam() {};
    PinholeCam(int _cam_width,
               int _cam_height,
               double _fx,
               double _fy,
               double _cx,
               double _cy,
               double _k1,
               double _k2,
               double _k3,
               double _r1,
               double _r2) :
        cam_width(_cam_width),
        cam_height(_cam_height),
        fx(_fx),
        fy(_fy),
        cx(_cx),
        cy(_cy),
        k1(_k1),
        k2(_k2),
        k3(_k3),
        r1(_r1),
        r2(_r2)
    {
        cameraMatrix = cv::Mat::zeros(3, 3, cv::DataType<double>::type);
        cameraMatrix.at<double>(0, 0) = fx;
        cameraMatrix.at<double>(1, 1) = fy;
        cameraMatrix.at<double>(2, 2) = 1.0;
        cameraMatrix.at<double>(0, 2) = cx;
        cameraMatrix.at<double>(1, 2) = cy;

        distCoeffs = cv::Mat::zeros(1, 5, cv::DataType<double>::type);
        distCoeffs.at<double>(0) = k1;
        distCoeffs.at<double>(1) = k2;
        distCoeffs.at<double>(2) = k3;
        distCoeffs.at<double>(3) = r1;
        distCoeffs.at<double>(4) = r2;

        cv::initUndistortRectifyMap(
            cameraMatrix,
            distCoeffs,
            cv::Mat_<double>::eye(3, 3),
            cameraMatrix,
            cv::Size(cam_width, cam_height),
            CV_32FC1,
            undist_map1, undist_map2);

        cout << "camera info loaded:"
             << "\ncam_width:\n" << cam_width
             << "\ncam_height:\n" << cam_height
             << "\ncameraMatrix:\n" << cameraMatrix
             << "\ndistCoeffs:\n" << distCoeffs
             << endl;
    }

    void
    undistort(cv::Mat &imgIn, cv::Mat &imgOut)
    {
        cv::remap(imgIn,
                  imgOut,
                  undist_map1,
                  undist_map2,
                  cv::INTER_LINEAR,
                  cv::BORDER_CONSTANT);
    }

    Eigen::Vector2d
    imgPt2norm(Eigen::Vector2d &imgPt)
    {
        Eigen::Vector2d normPt;
        normPt.x() = (imgPt.x() - cx) / fx;
        normPt.y() = (imgPt.y() - cy) / fy;
        return normPt;
    }

    Eigen::Vector2d
    normPt2img(Eigen::Vector3d normPt)
    {
        return Eigen::Vector2d(normPt.x() * fx + cx, normPt.y() * fy + cy);
    }

    Eigen::Vector2d
    Pt2img(Eigen::Vector3d Pt)
    {
        return Eigen::Vector2d(
            cvRound((Pt.x() / Pt.z()) * fx + cx),
            cvRound((Pt.y() / Pt.z()) * fy + cy));
    }

    bool
    is_in_FoV(Eigen::Vector2d &pt)
    {
        if (pt.x() < 0 || pt.y() < 0 ||
            pt.x() >= cam_width || pt.y() >= cam_height
            )
            return false;
        else
            return true;
    }

    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;

    int cam_width;
    int cam_height;

private:


    double fx;
    double fy;
    double cx;
    double cy;
    double k1;
    double k2;
    double k3;
    double r1;
    double r2;

    cv::Mat undist_map1;
    cv::Mat undist_map2;

};


#endif //CHECKERBOARD_LC_CALIB_PINHOLECAM_H
