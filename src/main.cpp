#include "GBoW.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <GSLAM/core/GSLAM.h>
#include <GSLAM/core/Timer.h>
#include "gtest.h"

using std::ifstream;

void createVocabulary(void* ,std::string ,std::string sParams)
{
    ifstream ifs(sParams.c_str());
    if(!ifs.is_open()) return ;

    cv::Ptr<cv::Feature2D> feature;

    std::string featureType=svar.GetString("Vocabulary.Feature","ORB");
    if(featureType=="ORB") feature=new cv::ORB();
    else if(featureType=="SIFT") feature=new cv::SIFT();

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

    std::cout<<"Creating vocabulary from image features.\n";


    SPtr<GSLAM::Vocabulary> vocabulary=GSLAM::Vocabulary::create(features,10,4);

    std::cout<<(*vocabulary)<<std::endl;

    std::string vocabularyfile2save=svar.GetString("Vocabulary.Save","vocabulary.gbow");
    if(vocabularyfile2save.size())
    {
        GSLAM::ScopedTimer timerSaveVocabulary("SaveVocabulary");
        vocabulary->save(vocabularyfile2save);
    }
}

void loadVocabulary(void* ,std::string ,std::string sParams)
{
    SCOPE_TIMER
    GSLAM::Vocabulary vocabulary(sParams);

    std::cout<<"Loaded vocabulary from "<<sParams<<"\n"<<vocabulary<<std::endl;

    std::string vocabularyfile2save=svar.GetString("Vocabulary.Save","vocabulary.gbow");
    if(sParams!=vocabularyfile2save)
    {
        GSLAM::ScopedTimer timerSaveVocabulary("SaveVocabulary");
        std::cout<<"Saving vocabulary to "<<vocabularyfile2save<<std::endl;
        vocabulary.save(vocabularyfile2save);
    }
}

int main(int argc,char** argv)
{
    scommand.RegisterCommand("createVocabulary",createVocabulary,NULL);
    scommand.RegisterCommand("loadVocabulary",loadVocabulary,NULL);
    svar.ParseMain(argc,argv);

    testing::InitGoogleTest(&argc,argv);

    return RUN_ALL_TESTS();
}
