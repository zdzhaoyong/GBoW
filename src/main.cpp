#include "GBoW.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <GSLAM/core/GSLAM.h>

using std::ifstream;

void createVocabulary(void* ,std::string ,std::string sParams)
{
    ifstream ifs(sParams.c_str());
    if(!ifs.is_open()) return ;

    cv::Ptr<cv::Feature2D> feature;

    feature=new cv::ORB();

    std::vector<GSLAM::TinyMat> features;
    std::string line;
    while(std::getline(ifs,line))
    {
        cv::Mat img=cv::imread(line);
        cv::Mat descriptors;
        std::vector<cv::KeyPoint> keypoints;
        if(img.empty()) continue;
        (*feature)(img,cv::Mat(),keypoints,descriptors);

        std::cout<<line<<","<<keypoints.size()<<std::endl;

        if(!descriptors.empty()) features.push_back(GSLAM::GImage(descriptors));
    }
    if(features.empty()) return ;

    GSLAM::Vocabulary vocabulary(10,3);
    std::cout<<"Creating vocabulary from image features.\n"<<vocabulary<<std::endl;

    vocabulary.create(features);

    std::cout<<vocabulary;

}

int main(int argc,char** argv)
{
    scommand.RegisterCommand("createVocabulary",createVocabulary,NULL);
    svar.ParseMain(argc,argv);

    return 0;
}
