#include "Vocabulary.h"
#include <list>

namespace GSLAM {

//class KeyFrameDatabase
//{
//public:
//    KeyFrameDatabase():mpVoc(NULL),mMap(NULL){}
//    KeyFrameDatabase(ORBVocabulary* voc,Map* global_map);

//   void add(FrameID pKF);

//   void erase(FrameID pKF);

//   void clear();

//   // Neighbor node
//   std::vector<FrameID> DetectNeighborCandidates(FrameID pKF, float minWords, int useAccumulateScore=0);

//   // Loop Detection
//   std::vector<FrameID> DetectLoopCandidates(FrameID pKF, float minScore, int useAccumulateScore=0);

//   // Relocalisation
//   std::vector<FrameID> DetectRelocalisationCandidates(BOW_Object* F);

//protected:

//  // Associated vocabulary
//  const ORBVocabulary* mpVoc;

//  Map* mMap;

//  // Inverted file
//  std::vector<list<FrameID> > mvInvertedFile;


//  // Mutex
//  pi::Mutex mMutex;
//};


}
