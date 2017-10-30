#include "Vocabulary.h"
#include "DBoW3/Vocabulary.h"
#include "DBoW3/DescManip.h"
#include "gtest.h"
#include <GSLAM/core/Svar.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;

TEST(Vocabulary,train)
{
    std::string trainfile=svar.GetString("TrainFile","images.txt");
    ifstream ifs(trainfile.c_str());
    if(!ifs.is_open()) return ;

    cv::Ptr<cv::Feature2D> feature;

    std::string featureType=svar.GetString("Vocabulary.Feature","ORB");
    if(featureType=="ORB") feature=new cv::ORB();

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


    SPtr<GSLAM::Vocabulary> vocptr=GSLAM::Vocabulary::create(features,10,4);
    GSLAM::Vocabulary& voc=*vocptr;

    DBoW3::Vocabulary vocref(10,4);
    std::vector<cv::Mat> mats;
    for(GSLAM::GImage feature:features) mats.push_back(feature);
    vocref.create(mats);


}

TEST(Vocabulary,loadsave)
{
    std::string vocfile=svar.GetString("Vocabulary.Load","/data/zhaoyong/Program/Thirdparty/ORB_SLAM2/Vocabulary/ORBvoc.txt");
    GSLAM::Vocabulary vocref(vocfile);

    if(vocfile.find(".gbow")!=std::string::npos) return;
    vocref.save("voctemp.gbow");
    GSLAM::Vocabulary voc("voctemp.gbow");

    EXPECT_EQ(voc.size(),vocref.size());
    EXPECT_EQ(voc.getScoringType(),vocref.getScoringType());
    EXPECT_EQ(voc.getWeightingType(),vocref.getWeightingType());
    EXPECT_EQ(voc.getDepthLevels(),vocref.getDepthLevels());
    EXPECT_EQ(voc.getBranchingFactor(),vocref.getBranchingFactor());
    EXPECT_EQ(voc.getEffectiveLevels(),vocref.getEffectiveLevels());
    EXPECT_EQ(voc.getDescritorSize(),vocref.getDescritorSize());
    EXPECT_EQ(voc.getDescritorType(),vocref.getDescritorType());
    EXPECT_EQ(voc.m_nodes.size(),vocref.m_nodes.size());

    for(int i=0;i<voc.size();i++)
    {
        GSLAM::Vocabulary::Node& node=*voc.m_words[i];
        GSLAM::Vocabulary::Node& node1=*vocref.m_words[i];
        EXPECT_EQ(node.id,node1.id);
    }

    for(int i=0;i<voc.m_nodes.size();i++)
    {
        GSLAM::Vocabulary::Node& node=voc.m_nodes[i];
        GSLAM::Vocabulary::Node& node1=vocref.m_nodes[i];
        EXPECT_EQ(node.id,node1.id);
        EXPECT_EQ(node.parent,node1.parent);
        EXPECT_EQ(node.weight,node1.weight);
        EXPECT_EQ(node.word_id,node1.word_id);
        EXPECT_EQ(node.childNum,node1.childNum);
        for(int j=0;j<node.childNum;j++)
        {
            EXPECT_EQ(node.child[j],node1.child[j]);
        }
        EXPECT_FALSE(memcmp(node.descriptor.data,node1.descriptor.data,
                            node1.descriptor.total()*node1.descriptor.elemSize()));
    }

    // distance
    cv::Mat img=cv::imread(svar.GetString("Image","/data/zhaoyong/Media/Photo/2005/20050209春节游中心公园/Dscf0016.jpg"));
    EXPECT_FALSE(img.empty());

    cv::Ptr<cv::Feature2D> feature;

    std::string featureType=svar.GetString("Vocabulary.Feature","ORB");
    if(featureType=="ORB") feature=new cv::ORB();

    std::vector<cv::KeyPoint> kps;
    cv::Mat des;
    (*feature)(img,cv::Mat(),kps,des);

    EXPECT_FALSE(des.empty());

    EXPECT_EQ(voc.distance(des.row(0),des.row(1)),DBoW3::DescManip::distance_8uc1(des.row(0),des.row(1)));

    cv::Mat row=des.row(0);
    GSLAM::WordId wid;
    GSLAM::WordValue wv;
    voc.transform(row,wid,wv);

    GSLAM::WordId wid1;
    GSLAM::WordValue wv1;
    vocref.transform(row,wid1,wv1);

    EXPECT_EQ(wid,wid1);
    EXPECT_EQ(wv,wv1);

    GSLAM::NodeId nid;
    GSLAM::NodeId nid1;
    voc.transform(row,wid,wv,&nid,4);
    vocref.transform(row,wid1,wv1,&nid1,4);
    EXPECT_EQ(wid,wid1);
    EXPECT_EQ(wv,wv1);
    EXPECT_EQ(nid,nid1);

    std::vector<GSLAM::GImage> desvec;
    std::vector<GSLAM::GImage>       desvec1;
    desvec.reserve(des.rows);
    desvec1.reserve(des.rows);
    for(int i=0;i<des.rows;i++){
        desvec.push_back(des.row(i));
        desvec1.push_back(des.row(i));
    }

    GSLAM::BowVector bowvec;
    GSLAM::FeatureVector featvec;

    GSLAM::BowVector bowvec1;
    GSLAM::FeatureVector featvec1;
    voc.transform(desvec,bowvec,featvec,4);
    vocref.transform(desvec1,bowvec1,featvec1,4);

    EXPECT_EQ(bowvec.size(),bowvec1.size());
    EXPECT_EQ(featvec.size(),featvec1.size());

    for(GSLAM::BowVector::value_type v:bowvec)
    {
        EXPECT_NEAR(v.second,bowvec1[v.first],0.001);
    }
    for(GSLAM::FeatureVector::value_type v:featvec)
    {
        EXPECT_EQ(v.second,featvec1[v.first]);
    }

}

TEST(Vocabulary,functional)
{
    std::string vocfile=svar.GetString("Vocabulary.Load","/data/zhaoyong/Program/Thirdparty/ORB_SLAM2/Vocabulary/ORBvoc.txt");
    if(vocfile.empty()) return ;
    GSLAM::Vocabulary voc(vocfile);
    EXPECT_FALSE(voc.empty());

    DBoW3::Vocabulary vocref(vocfile);
    EXPECT_EQ(voc.size(),vocref.size());
    EXPECT_EQ(voc.getScoringType(),vocref.getScoringType());
    EXPECT_EQ(voc.getWeightingType(),vocref.getWeightingType());
    EXPECT_EQ(voc.getDepthLevels(),vocref.getDepthLevels());
    EXPECT_EQ(voc.getBranchingFactor(),vocref.getBranchingFactor());
    EXPECT_EQ(voc.getEffectiveLevels(),vocref.getEffectiveLevels());
    EXPECT_EQ(voc.getDescritorSize(),vocref.getDescritorSize());
    EXPECT_EQ(voc.getDescritorType(),vocref.getDescritorType());
    EXPECT_NE(voc.m_nodes.size(),0);
    EXPECT_FALSE(voc.m_nodeDescriptors.empty());

    for(int i=0;i<voc.size();i++)
    {
        EXPECT_TRUE(memcmp(voc.getWord(i).data,vocref.getWord(i).data,voc.getDescritorSize())==0);
    }

    // distance
    cv::Mat img=cv::imread(svar.GetString("Image","/data/zhaoyong/Media/Photo/2005/20050209春节游中心公园/Dscf0016.jpg"));
    EXPECT_FALSE(img.empty());

    cv::Ptr<cv::Feature2D> feature;

    std::string featureType=svar.GetString("Vocabulary.Feature","ORB");
    if(featureType=="ORB") feature=new cv::ORB();

    std::vector<cv::KeyPoint> kps;
    cv::Mat des;
    (*feature)(img,cv::Mat(),kps,des);

    EXPECT_FALSE(des.empty());

    EXPECT_EQ(voc.distance(des.row(0),des.row(1)),DBoW3::DescManip::distance_8uc1(des.row(0),des.row(1)));

    cv::Mat row=des.row(0);
    GSLAM::WordId wid;
    GSLAM::WordValue wv;
    voc.transform(row,wid,wv);

    DBoW3::WordId wid1;
    DBoW3::WordValue wv1;
    vocref.transform(row,wid1,wv1);

    EXPECT_EQ(wid,wid1);
    EXPECT_EQ(wv,wv1);

    GSLAM::NodeId nid;
    DBoW3::NodeId nid1;
    voc.transform(row,wid,wv,&nid,4);
    vocref.transform(row,wid1,wv1,&nid1,4);
    EXPECT_EQ(wid,wid1);
    EXPECT_EQ(wv,wv1);
    EXPECT_EQ(nid,nid1);

    std::vector<GSLAM::GImage> desvec;
    std::vector<cv::Mat>       desvec1;
    desvec.reserve(des.rows);
    desvec1.reserve(des.rows);
    for(int i=0;i<des.rows;i++){
        desvec.push_back(des.row(i));
        desvec1.push_back(des.row(i));
    }

    GSLAM::BowVector bowvec;
    GSLAM::FeatureVector featvec;

    DBoW3::BowVector bowvec1;
    DBoW3::FeatureVector featvec1;
    voc.transform(desvec,bowvec,featvec,4);
    vocref.transform(desvec1,bowvec1,featvec1,4);

    EXPECT_EQ(bowvec.size(),bowvec1.size());
    EXPECT_EQ(featvec.size(),featvec1.size());

    for(GSLAM::BowVector::value_type v:bowvec)
    {
        EXPECT_NEAR(v.second,bowvec1[v.first],0.001);
    }
    for(GSLAM::FeatureVector::value_type v:featvec)
    {
        EXPECT_EQ(v.second,featvec1[v.first]);
    }
}
