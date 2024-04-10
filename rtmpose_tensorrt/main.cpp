#include <iostream>
#include <string>
#include <tuple>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>

#include "rtmdet.h"
#include "rtmpose.h"
#include "utils.h"
#include "inference.h"


using namespace std;

/**
 * @brief Setting up Tensorrt logger
*/
class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // Only output logs with severity greater than warning
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
}logger;


int main()
{
    // Setting inference mode: 0 represents image, 1 represents camera, and 2 represents video
    int mode = 1;

    // set engine file path
    string detEngineFile = "./model/rtmdet.engine";
    string poseEngineFile = "./model/rtmpose_s.engine";

    // set iou and conf threshold
    float iou_threshold = 0.65;
    float conf_threshold = 0.5;

    // init model
    RTMDet det_model(detEngineFile, logger, conf_threshold, iou_threshold);
    RTMPose pose_model(poseEngineFile, logger);

    if (mode == 0)
    {
        // get image path
        string image_path;

        cout << "Please enter the path of the image:" << endl;
        cin >> image_path;

        cv::Mat image = cv::imread(image_path);
        if (image.empty())
        {
            cout << "Image not found: " << image_path << endl;
            exit(0);
        }

        cv::Mat show_image;
        image.copyTo(show_image);

        // inference the image
        auto result = inference(image, det_model, pose_model);
        draw_pose(show_image, result);

        // show image
        cv::imshow("result", show_image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    else if (mode == 1)
    {
        // open cap
        cv::VideoCapture cap(0);

        if (!cap.isOpened())
        {
            cout << "Camera not found." << endl;
            exit(0);
        }

        while (cap.isOpened())
        {
            cv::Mat frame;
            cv::Mat show_frame;
            cap >> frame;

            if (frame.empty())
                break;

            frame.copyTo(show_frame);
            auto result = inference(frame, det_model, pose_model);
            draw_pose(show_frame, result);

            cv::imshow("result", show_frame);
            if (cv::waitKey(1) == 'q')
                break;
        }
        cv::destroyAllWindows();
        cap.release();
    }
    else if (mode == 2)
    {   
        // get video path
        string video_path;

        cout << "Please enter the path of the video:" << endl;
        cin >> video_path;

        // open cap
        cv::VideoCapture cap(video_path);

        if (!cap.isOpened())
        {
            cout << "Video not found: " << video_path << endl;
            exit(0);
        }

        while (cap.isOpened())
        {
            cv::Mat frame;
            cv::Mat show_frame;
            cap >> frame;

            if (frame.empty())
                break;

            frame.copyTo(show_frame);
            auto result = inference(frame, det_model, pose_model);
            draw_pose(show_frame, result);

            cv::imshow("result", show_frame);
            if (cv::waitKey(1) == 'q')
                break;
        }
        cv::destroyAllWindows();
        cap.release();
    }

    return 0;
}

