#ifndef GSLAM_VOCABULARY_H
#define GSLAM_VOCABULARY_H

#include <stdlib.h>
#include <map>
#include <vector>
#include <cfloat>
#include <cmath>
#include <string>
#include <sstream>
#include <iostream>
#include <stdint.h>
#include <limits.h>
#include <numeric>      // std::accumulate
#include <chrono>

#include <GSLAM/core/GImage.h>
#include <GSLAM/core/SPtr.h>

#define GSLAM_VOCABULARY_KMAX 10

namespace GSLAM {

typedef size_t                 NodeId;
typedef size_t                 WordId;
typedef float                  WordValue;
typedef std::map<WordId,float> BowVector;
typedef std::map<WordId,std::vector<unsigned int> > FeatureVector;

class TinyMat: public GImage
{
public:
    TinyMat(){}

    TinyMat(const GImage& gimage):GImage(gimage){}

    TinyMat(int rows_,int cols_,int type=GImageType<>::Type,uchar* src=NULL,bool copy=true)
        :GImage(cols_,rows_,type,src,copy){}

    template <typename T>
    T* ptr()const{return (T*)data;}

    template <typename T>
    T* ptr(const int& idx)const{return ((T*)data)+idx;}

    const TinyMat row(int idx=0)const{return TinyMat(1,cols,type(),data+elemSize()*cols*idx);}

};

class GeneralScoring;
class Vocabulary
{
public:
    friend class GBoW;


    /// L-norms for normalization
    enum LNorm
    {
      L1,
      L2
    };

    /// Weighting type
    enum WeightingType
    {
      TF_IDF,
      TF,
      IDF,
      BINARY
    };

    /// Scoring type
    enum ScoringType
    {
      L1_NORM,
      L2_NORM,
      CHI_SQUARE,
      KL,
      BHATTACHARYYA,
      DOT_PRODUCT
    };


    /**
     * Initiates an empty vocabulary
     * @param k branching factor
     * @param L depth levels
     * @param weighting weighting type
     * @param scoring scoring type
     */
    Vocabulary(int k = 10, int L = 5,
      WeightingType weighting = TF_IDF, ScoringType scoring = L1_NORM);

    /**
     * Creates the vocabulary by loading a file
     * @param filename
     */
    Vocabulary(const std::string &filename);

    /**
     * Saves the vocabulary into a file. If filename extension contains .yml, opencv YALM format is used. Otherwise, binary format is employed
     * @param filename
     */
    void save(const std::string &filename, bool binary_compressed=true) const;

    /**
     * Loads the vocabulary from a file created with save
     * @param filename.
     */
    void load(const std::string &filename);
    /**
     * Creates a vocabulary from the training features with the already
     * defined parameters
     * @param training_features
     */
    virtual void create
      (const std::vector<std::vector<TinyMat> > &training_features);
    /**
     * Creates a vocabulary from the training features with the already
     * defined parameters
     * @param training_features. Each row of a matrix is a feature
     */
     virtual void create
      (const  std::vector<TinyMat>   &training_features);

    /**
     * Creates a vocabulary from the training features, setting the branching
     * factor and the depth levels of the tree
     * @param training_features
     * @param k branching factor
     * @param L depth levels
     */
    virtual void create
      (const std::vector<std::vector<TinyMat> > &training_features,
        int k, int L);

    /**
     * Creates a vocabulary from the training features, setting the branching
     * factor nad the depth levels of the tree, and the weighting and scoring
     * schemes
     */
    virtual void create
      (const std::vector<std::vector<TinyMat> > &training_features,
        int k, int L, WeightingType weighting, ScoringType scoring);
    /**
     * Returns the number of words in the vocabulary
     * @return number of words
     */
    virtual inline unsigned int size() const{  return (unsigned int)m_words.size();}


    /**
     * Returns whether the vocabulary is empty (i.e. it has not been trained)
     * @return true iff the vocabulary is empty
     */
    virtual inline bool empty() const{ return m_words.empty();}

    /** Clears the vocabulary object
     */
    void clear();

    void transform(const std::vector<TinyMat>& features, BowVector &v)const;
    /**
     * Transforms a set of descriptores into a bow vector
     * @param features, one per row
     * @param v (out) bow vector of weighted words
     */
    virtual void transform(const  TinyMat & features, BowVector &v)
      const;
    /**
     * Transform a set of descriptors into a bow vector and a feature vector
     * @param features
     * @param v (out) bow vector
     * @param fv (out) feature vector of nodes and feature indexes
     * @param levelsup levels to go up the vocabulary tree to get the node index
     */
    virtual void transform(const std::vector<TinyMat>& features,
      BowVector &v, FeatureVector &fv, int levelsup) const;

    /**
     * Transforms a single feature into a word (without weight)
     * @param feature
     * @return word id
     */
    virtual WordId transform(const TinyMat& feature) const;

    /**
     * Returns the score of two vectors
     * @param a vector
     * @param b vector
     * @return score between vectors
     * @note the vectors must be already sorted and normalized if necessary
     */
    inline double score(const BowVector &a, const BowVector &b) const;

    /**
     * Returns the id of the node that is "levelsup" levels from the word given
     * @param wid word id
     * @param levelsup 0..L
     * @return node id. if levelsup is 0, returns the node id associated to the
     *   word id
     */
    virtual NodeId getParentNode(WordId wid, int levelsup) const;

    /**
     * Returns the ids of all the words that are under the given node id,
     * by traversing any of the branches that goes down from the node
     * @param nid starting node id
     * @param words ids of words
     */
    void getWordsFromNode(NodeId nid, std::vector<WordId> &words) const;

    /**
     * Returns the branching factor of the tree (k)
     * @return k
     */
    inline int getBranchingFactor() const { return m_k; }

    /**
     * Returns the depth levels of the tree (L)
     * @return L
     */
    inline int getDepthLevels() const { return m_L; }

    /**
     * Returns the real depth levels of the tree on average
     * @return average of depth levels of leaves
     */
    float getEffectiveLevels() const;

    /**
     * Returns the descriptor of a word
     * @param wid word id
     * @return descriptor
     */
    virtual inline TinyMat getWord(WordId wid) const;

    /**
     * Returns the weight of a word
     * @param wid word id
     * @return weight
     */
    virtual inline WordValue getWordWeight(WordId wid) const;

    /**
     * Returns the weighting method
     * @return weighting method
     */
    inline WeightingType getWeightingType() const { return m_weighting; }

    /**
     * Returns the scoring method
     * @return scoring method
     */
    inline ScoringType getScoringType() const { return m_scoring; }

    /**
     * Changes the weighting method
     * @param type new weighting type
     */
    inline void setWeightingType(WeightingType type);

    /**
     * Changes the scoring method
     * @param type new scoring type
     */
    void setScoringType(ScoringType type);


    /**
     * Stops those words whose weight is below minWeight.
     * Words are stopped by setting their weight to 0. There are not returned
     * later when transforming image features into vectors.
     * Note that when using IDF or TF_IDF, the weight is the idf part, which
     * is equivalent to -log(f), where f is the frequency of the word
     * (f = Ni/N, Ni: number of training images where the word is present,
     * N: number of training images).
     * Note that the old weight is forgotten, and subsequent calls to this
     * function with a lower minWeight have no effect.
     * @return number of words stopped now
     */
    virtual int stopWords(double minWeight);


    /** Returns the size of the descriptor employed. If the Vocabulary is empty, returns -1
     */
    int getDescritorSize()const{return m_nodeDescriptors.cols;}
    /** Returns the type of the descriptor employed normally(8U_C1, 32F_C1)
     */
    int getDescritorType()const{return m_nodeDescriptors.type();}


    /**
     * Calculates the mean value of a set of descriptors
     * @param descriptors
     * @param mean mean descriptor
     */
     static void meanValue(const std::vector<TinyMat> &descriptors,
      TinyMat &mean)  ;

    /**
     * Calculates the distance between two descriptors
     * @param a
     * @param b
     * @return distance
     */
     static double distance(const TinyMat &a, const TinyMat &b);
     static  inline uint32_t distance_8uc1(const TinyMat &a, const TinyMat &b);

protected:

  /**
   * Creates an instance of the scoring object accoring to m_scoring
   */
  void createScoringObject();

  /**
   * Returns a set of pointers to descriptores
   * @param training_features all the features
   * @param features (out) pointers to the training features
   */
  void getFeatures(const std::vector<std::vector<TinyMat> > &training_features,
    std::vector<TinyMat> &features) const;

  /**
   * Returns the word id associated to a feature
   * @param feature
   * @param id (out) word id
   * @param weight (out) word weight
   * @param nid (out) if given, id of the node "levelsup" levels up
   * @param levelsup
   */
  virtual void transform(const TinyMat &feature,
    WordId &id, WordValue &weight, NodeId* nid  , int levelsup = 0) const;
  /**
   * Returns the word id associated to a feature
   * @param feature
   * @param id (out) word id
   * @param weight (out) word weight
   * @param nid (out) if given, id of the node "levelsup" levels up
   * @param levelsup
   */
  virtual void transform(const TinyMat &feature,
    WordId &id, WordValue &weight ) const;

  /**
   * Returns the word id associated to a feature
   * @param feature
   * @param id (out) word id
   */
  virtual void transform(const TinyMat &feature, WordId &id) const;

  static  void addWeight(BowVector& bow,WordId id, WordValue v)
  {
      BowVector::iterator vit = bow.lower_bound(id);

      if(vit != bow.end() && !(bow.key_comp()(id, vit->first)))
      {
        vit->second += v;
      }
      else
      {
        bow.insert(vit, BowVector::value_type(id, v));
      }
  }

  static void addIfNotExist(BowVector& bow,WordId id, WordValue v)
  {
      BowVector::iterator vit = bow.lower_bound(id);

      if(vit == bow.end() || (bow.key_comp()(id, vit->first)))
      {
        bow.insert(vit, BowVector::value_type(id, v));
      }

  }

  static void normalize(BowVector& bow,LNorm norm_type)
  {
      double norm = 0.0;
      BowVector::iterator it;

      if(norm_type == L1)
      {
        for(it = bow.begin(); it != bow.end(); ++it)
          norm += fabs(it->second);
      }
      else
      {
        for(it = bow.begin(); it != bow.end(); ++it)
          norm += it->second * it->second;
                    norm = sqrt(norm);
      }

      if(norm > 0.0)
      {
        for(it = bow.begin(); it != bow.end(); ++it)
          it->second /= norm;
      }
  }

  static void addFeature(FeatureVector& fvec,NodeId id, unsigned int i_feature)
  {
      FeatureVector::iterator vit = fvec.lower_bound(id);

      if(vit != fvec.end() && vit->first == id)
      {
        vit->second.push_back(i_feature);
      }
      else
      {
        vit = fvec.insert(vit, FeatureVector::value_type(id,
          std::vector<unsigned int>() ));
        vit->second.push_back(i_feature);
      }
  }

  /**
   * Creates a level in the tree, under the parent, by running kmeans with
   * a descriptor set, and recursively creates the subsequent levels too
   * @param parent_id id of parent node
   * @param descriptors descriptors to run the kmeans on
   * @param current_level current level in the tree
   */
  void HKmeansStep(NodeId parent_id, const std::vector<TinyMat> &descriptors,
    int current_level);

  /**
   * Creates k clusters from the given descriptors with some seeding algorithm.
   * @note In this class, kmeans++ is used, but this function should be
   *   overriden by inherited classes.
   */
  virtual void initiateClusters(const std::vector<TinyMat> &descriptors,
    std::vector<TinyMat> &clusters) const;

  /**
   * Creates k clusters from the given descriptor sets by running the
   * initial step of kmeans++
   * @param descriptors
   * @param clusters resulting clusters
   */
  void initiateClustersKMpp(const std::vector<TinyMat> &descriptors,
    std::vector<TinyMat> &clusters) const;

  /**
   * Create the words of the vocabulary once the tree has been built
   */
  void createWords();

  /**
   * Sets the weights of the nodes of tree according to the given features.
   * Before calling this function, the nodes and the words must be already
   * created (by calling HKmeansStep and createWords)
   * @param features
   */
  void setNodeWeights(const std::vector<std::vector<TinyMat> > &features);


  /// Tree node
  struct Node
  {
      /// Node id
      NodeId id;
      /// Weight if the node is a word
      WordValue weight;
      /// Parent node (undefined in case of root)
      NodeId parent;

      /// Children
      int          childNum;
      NodeId       child[GSLAM_VOCABULARY_KMAX];

      /// Word id if the node is a word
      WordId word_id;

      TinyMat descriptor;

      /**
       * Empty constructor
       */
      Node(): id(0), weight(0), parent(0), word_id(0),childNum(0){}

      /**
       * Constructor
       * @param _id node id
       */
      Node(NodeId _id): id(_id), weight(0), parent(0), word_id(0),childNum(0){}

      /**
       * Returns whether the node is a leaf node
       * @return true iff the node is a leaf
       */
      inline bool isLeaf() const { return childNum==0; }

      bool addChild(NodeId _id)
      {
          if(childNum>9) return false;
          child[childNum]=_id;
          childNum++;
          return true;
      }
  };

    /// Branching factor
    int m_k;

    /// Depth levels
    int m_L;

    /// Weighting method
    WeightingType m_weighting;

    /// Scoring method
    ScoringType m_scoring;

    /// Object for computing scores
    SPtr<GeneralScoring> m_scoring_object;

    /// Tree nodes
    std::vector<Node> m_nodes;
    TinyMat            m_nodeDescriptors;

    /// Words of the vocabulary (tree leaves)
    /// this condition holds: m_words[wid]->word_id == wid
    std::vector<Node*> m_words;
};
/// Base class of scoring functions
class GeneralScoring
{
public:
  /**
   * Computes the score between two vectors. Vectors must be sorted and
   * normalized if necessary
   * @param v (in/out)
   * @param w (in/out)
   * @return score
   */
  virtual double score(const BowVector &v, const BowVector &w) const = 0;

  /**
   * Returns whether a vector must be normalized before scoring according
   * to the scoring scheme
   * @param norm norm to use
   * @return true iff must normalize
   */
  virtual bool mustNormalize(Vocabulary::LNorm &norm) const = 0;

  /// Log of epsilon
        static const double LOG_EPS;
  // If you change the type of WordValue, make sure you change also the
        // epsilon value (this is needed by the KL method)

  virtual ~GeneralScoring() {} //!< Required for virtual base classes

};

/**
 * Macro for defining Scoring classes
 * @param NAME name of class
 * @param MUSTNORMALIZE if vectors must be normalized to compute the score
 * @param NORM type of norm to use when MUSTNORMALIZE
 */
#define __SCORING_CLASS(NAME, MUSTNORMALIZE, NORM) \
  NAME: public GeneralScoring \
  { public: \
    /** \
     * Computes score between two vectors \
     * @param v \
     * @param w \
     * @return score between v and w \
     */ \
    virtual double score(const BowVector &v, const BowVector &w) const; \
    \
    /** \
     * Says if a vector must be normalized according to the scoring function \
     * @param norm (out) if true, norm to use
     * @return true iff vectors must be normalized \
     */ \
    virtual inline bool mustNormalize(Vocabulary::LNorm &norm) const  \
      { norm = NORM; return MUSTNORMALIZE; } \
  }

/// L1 Scoring object
class __SCORING_CLASS(L1Scoring, true, Vocabulary::L1);

/// L2 Scoring object
class __SCORING_CLASS(L2Scoring, true, Vocabulary::L2);

/// Chi square Scoring object
class __SCORING_CLASS(ChiSquareScoring, true, Vocabulary::L1);

/// KL divergence Scoring object
class __SCORING_CLASS(KLScoring, true, Vocabulary::L1);

/// Bhattacharyya Scoring object
class __SCORING_CLASS(BhattacharyyaScoring, true, Vocabulary::L1);

/// Dot product Scoring object
class __SCORING_CLASS(DotProductScoring, false, Vocabulary::L1);

#undef __SCORING_CLASS
// If you change the type of WordValue, make sure you change also the
// epsilon value (this is needed by the KL method)
const double GeneralScoring::LOG_EPS = log(DBL_EPSILON); // FLT_EPSILON

uint32_t Vocabulary::distance_8uc1(const TinyMat &a, const TinyMat &b){
    //binary descriptor

        // Bit count function got from:
         // http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetKernighan
         // This implementation assumes that a.cols (CV_8U) % sizeof(uint64_t) == 0

         const uint64_t *pa=(uint64_t*)a.data, *pb=(uint64_t*)b.data;

         uint64_t v, ret = 0;
         int n=a.cols / sizeof(uint64_t);
         for(size_t i = 0; i < n; ++i, ++pa, ++pb)
         {
           v = *pa ^ *pb;
           v = v - ((v >> 1) & (uint64_t)~(uint64_t)0/3);
           v = (v & (uint64_t)~(uint64_t)0/15*3) + ((v >> 2) &
             (uint64_t)~(uint64_t)0/15*3);
           v = (v + (v >> 4)) & (uint64_t)~(uint64_t)0/255*15;
           ret += (uint64_t)(v * ((uint64_t)~(uint64_t)0/255)) >>
             (sizeof(uint64_t) - 1) * CHAR_BIT;
         }
         return ret;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double L1Scoring::score(const BowVector &v1, const BowVector &v2) const
{
  BowVector::const_iterator v1_it, v2_it;
  const BowVector::const_iterator v1_end = v1.end();
  const BowVector::const_iterator v2_end = v2.end();

  v1_it = v1.begin();
  v2_it = v2.begin();

  double score = 0;

  while(v1_it != v1_end && v2_it != v2_end)
  {
    const WordValue& vi = v1_it->second;
    const WordValue& wi = v2_it->second;

    if(v1_it->first == v2_it->first)
    {
      score += fabs(vi - wi) - fabs(vi) - fabs(wi);

      // move v1 and v2 forward
      ++v1_it;
      ++v2_it;
    }
    else if(v1_it->first < v2_it->first)
    {
      // move v1 forward
      v1_it = v1.lower_bound(v2_it->first);
      // v1_it = (first element >= v2_it.id)
    }
    else
    {
      // move v2 forward
      v2_it = v2.lower_bound(v1_it->first);
      // v2_it = (first element >= v1_it.id)
    }
  }

  // ||v - w||_{L1} = 2 + Sum(|v_i - w_i| - |v_i| - |w_i|)
  //		for all i | v_i != 0 and w_i != 0
  // (Nister, 2006)
  // scaled_||v - w||_{L1} = 1 - 0.5 * ||v - w||_{L1}
  score = -score/2.0;

  return score; // [0..1]
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double L2Scoring::score(const BowVector &v1, const BowVector &v2) const
{
  BowVector::const_iterator v1_it, v2_it;
  const BowVector::const_iterator v1_end = v1.end();
  const BowVector::const_iterator v2_end = v2.end();

  v1_it = v1.begin();
  v2_it = v2.begin();

  double score = 0;

  while(v1_it != v1_end && v2_it != v2_end)
  {
    const WordValue& vi = v1_it->second;
    const WordValue& wi = v2_it->second;

    if(v1_it->first == v2_it->first)
    {
      score += vi * wi;

      // move v1 and v2 forward
      ++v1_it;
      ++v2_it;
    }
    else if(v1_it->first < v2_it->first)
    {
      // move v1 forward
      v1_it = v1.lower_bound(v2_it->first);
      // v1_it = (first element >= v2_it.id)
    }
    else
    {
      // move v2 forward
      v2_it = v2.lower_bound(v1_it->first);
      // v2_it = (first element >= v1_it.id)
    }
  }

  // ||v - w||_{L2} = sqrt( 2 - 2 * Sum(v_i * w_i) )
        //		for all i | v_i != 0 and w_i != 0 )
        // (Nister, 2006)
        if(score >= 1) // rounding errors
          score = 1.0;
        else
    score = 1.0 - sqrt(1.0 - score); // [0..1]

  return score;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double ChiSquareScoring::score(const BowVector &v1, const BowVector &v2)
  const
{
  BowVector::const_iterator v1_it, v2_it;
  const BowVector::const_iterator v1_end = v1.end();
  const BowVector::const_iterator v2_end = v2.end();

  v1_it = v1.begin();
  v2_it = v2.begin();

  double score = 0;

  // all the items are taken into account

  while(v1_it != v1_end && v2_it != v2_end)
  {
    const WordValue& vi = v1_it->second;
    const WordValue& wi = v2_it->second;

    if(v1_it->first == v2_it->first)
    {
      // (v-w)^2/(v+w) - v - w = -4 vw/(v+w)
      // we move the -4 out
      if(vi + wi != 0.0) score += vi * wi / (vi + wi);

      // move v1 and v2 forward
      ++v1_it;
      ++v2_it;
    }
    else if(v1_it->first < v2_it->first)
    {
      // move v1 forward
      v1_it = v1.lower_bound(v2_it->first);
    }
    else
    {
      // move v2 forward
      v2_it = v2.lower_bound(v1_it->first);
    }
  }

  // this takes the -4 into account
  score = 2. * score; // [0..1]

  return score;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double KLScoring::score(const BowVector &v1, const BowVector &v2) const
{
  BowVector::const_iterator v1_it, v2_it;
  const BowVector::const_iterator v1_end = v1.end();
  const BowVector::const_iterator v2_end = v2.end();

  v1_it = v1.begin();
  v2_it = v2.begin();

  double score = 0;

  // all the items or v are taken into account

  while(v1_it != v1_end && v2_it != v2_end)
  {
    const WordValue& vi = v1_it->second;
    const WordValue& wi = v2_it->second;

    if(v1_it->first == v2_it->first)
    {
      if(vi != 0 && wi != 0) score += vi * log(vi/wi);

      // move v1 and v2 forward
      ++v1_it;
      ++v2_it;
    }
    else if(v1_it->first < v2_it->first)
    {
      // move v1 forward
      score += vi * (log(vi) - LOG_EPS);
      ++v1_it;
    }
    else
    {
      // move v2_it forward, do not add any score
      v2_it = v2.lower_bound(v1_it->first);
      // v2_it = (first element >= v1_it.id)
    }
  }

  // sum rest of items of v
  for(; v1_it != v1_end; ++v1_it)
    if(v1_it->second != 0)
      score += v1_it->second * (log(v1_it->second) - LOG_EPS);

  return score; // cannot be scaled
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double BhattacharyyaScoring::score(const BowVector &v1,
  const BowVector &v2) const
{
  BowVector::const_iterator v1_it, v2_it;
  const BowVector::const_iterator v1_end = v1.end();
  const BowVector::const_iterator v2_end = v2.end();

  v1_it = v1.begin();
  v2_it = v2.begin();

  double score = 0;

  while(v1_it != v1_end && v2_it != v2_end)
  {
    const WordValue& vi = v1_it->second;
    const WordValue& wi = v2_it->second;

    if(v1_it->first == v2_it->first)
    {
      score += sqrt(vi * wi);

      // move v1 and v2 forward
      ++v1_it;
      ++v2_it;
    }
    else if(v1_it->first < v2_it->first)
    {
      // move v1 forward
      v1_it = v1.lower_bound(v2_it->first);
      // v1_it = (first element >= v2_it.id)
    }
    else
    {
      // move v2 forward
      v2_it = v2.lower_bound(v1_it->first);
      // v2_it = (first element >= v1_it.id)
    }
  }

  return score; // already scaled
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double DotProductScoring::score(const BowVector &v1,
  const BowVector &v2) const
{
  BowVector::const_iterator v1_it, v2_it;
  const BowVector::const_iterator v1_end = v1.end();
  const BowVector::const_iterator v2_end = v2.end();

  v1_it = v1.begin();
  v2_it = v2.begin();

  double score = 0;

  while(v1_it != v1_end && v2_it != v2_end)
  {
    const WordValue& vi = v1_it->second;
    const WordValue& wi = v2_it->second;

    if(v1_it->first == v2_it->first)
    {
      score += vi * wi;

      // move v1 and v2 forward
      ++v1_it;
      ++v2_it;
    }
    else if(v1_it->first < v2_it->first)
    {
      // move v1 forward
      v1_it = v1.lower_bound(v2_it->first);
      // v1_it = (first element >= v2_it.id)
    }
    else
    {
      // move v2 forward
      v2_it = v2.lower_bound(v1_it->first);
      // v2_it = (first element >= v1_it.id)
    }
  }

  return score; // cannot scale
}

Vocabulary::Vocabulary
  (int k, int L, WeightingType weighting, ScoringType scoring)
  : m_k(k), m_L(L), m_weighting(weighting), m_scoring(scoring)
{
  createScoringObject();
}

// --------------------------------------------------------------------------


Vocabulary::Vocabulary
  (const std::string &filename): m_scoring_object(NULL)
{
  load(filename);
}

// --------------------------------------------------------------------------


void Vocabulary::createScoringObject()
{
  m_scoring_object.reset();

  switch(m_scoring)
  {
    case L1_NORM:
      m_scoring_object = SPtr<GeneralScoring>(new L1Scoring);
      break;

    case L2_NORM:
      m_scoring_object = SPtr<GeneralScoring>(new L2Scoring);
      break;

    case CHI_SQUARE:
      m_scoring_object = SPtr<GeneralScoring>(new ChiSquareScoring);
      break;

    case KL:
      m_scoring_object = SPtr<GeneralScoring>(new KLScoring);
      break;

    case BHATTACHARYYA:
      m_scoring_object = SPtr<GeneralScoring>(new BhattacharyyaScoring);
      break;

    case DOT_PRODUCT:
      m_scoring_object = SPtr<GeneralScoring>(new DotProductScoring);
      break;

  }
}

// --------------------------------------------------------------------------


void Vocabulary::setScoringType(ScoringType type)
{
  m_scoring = type;
  createScoringObject();
}

// --------------------------------------------------------------------------


void Vocabulary::setWeightingType(WeightingType type)
{
  this->m_weighting = type;
}

void Vocabulary::create(const std::vector< TinyMat > &training_features)
{
    std::vector<std::vector<TinyMat> > vtf(training_features.size());
    int colbytes=training_features[0].cols*training_features[0].elemSize();
    for(size_t i=0;i<training_features.size();i++){
        vtf[i].resize(training_features[i].rows);
        for(int r=0;r<training_features[i].rows;r++)
            vtf[i][r]=TinyMat(1,training_features[i].cols,training_features[i].type(),
                             training_features[i].data+r*colbytes,false);
    }
    create(vtf);

}

void Vocabulary::create(
  const std::vector<std::vector<TinyMat> > &training_features)
{
  m_nodes.clear();
  m_words.clear();

  // expected_nodes = Sum_{i=0..L} ( k^i )
    int expected_nodes =
        (int)((pow((double)m_k, (double)m_L + 1) - 1)/(m_k - 1));

  m_nodes.reserve(expected_nodes); // avoid allocations when creating the tree


  std::vector<TinyMat> features;
  getFeatures(training_features, features);


  // create root
  m_nodes.push_back(Node(0)); // root

  // create the tree
  HKmeansStep(0, features, 1);

  // create the words
  createWords();

  // and set the weight of each node of the tree
  setNodeWeights(training_features);

}

// --------------------------------------------------------------------------


void Vocabulary::create(
  const std::vector<std::vector<TinyMat> > &training_features,
  int k, int L)
{
  m_k = k;
  m_L = L;

  create(training_features);
}

// --------------------------------------------------------------------------


void Vocabulary::create(
  const std::vector<std::vector<TinyMat> > &training_features,
  int k, int L, WeightingType weighting, ScoringType scoring)
{
  m_k = k;
  m_L = L;
  m_weighting = weighting;
  m_scoring = scoring;
  createScoringObject();

  create(training_features);
}

// --------------------------------------------------------------------------


void Vocabulary::getFeatures(
  const std::vector<std::vector<TinyMat> > &training_features,
  std::vector<TinyMat> &features) const
{
  features.resize(0);
  for(size_t i=0;i<training_features.size();i++)
      for(size_t j=0;j<training_features[i].size();j++)
              features.push_back(training_features[i][j]);
}

// --------------------------------------------------------------------------


void Vocabulary::HKmeansStep(NodeId parent_id,
                             const std::vector<TinyMat> &descriptors, int current_level)
{

    if(descriptors.empty()) return;

    // features associated to each cluster
    std::vector<TinyMat> clusters;
    std::vector<std::vector<unsigned int> > groups; // groups[i] = [j1, j2, ...]
    // j1, j2, ... indices of descriptors associated to cluster i

    clusters.reserve(m_k);
    groups.reserve(m_k);


    if((int)descriptors.size() <= m_k)
    {
        // trivial case: one cluster per feature
        groups.resize(descriptors.size());

        for(unsigned int i = 0; i < descriptors.size(); i++)
        {
            groups[i].push_back(i);
            clusters.push_back(descriptors[i]);
        }
    }
    else
    {
        // select clusters and groups with kmeans

        bool first_time = true;
        bool goon = true;

        // to check if clusters move after iterations
        std::vector<int> last_association, current_association;

        while(goon)
        {
            // 1. Calculate clusters

            if(first_time)
            {
                // random sample
                initiateClusters(descriptors, clusters);
            }
            else
            {
                // calculate cluster centres

                for(unsigned int c = 0; c < clusters.size(); ++c)
                {
                    std::vector<TinyMat> cluster_descriptors;
                    cluster_descriptors.reserve(groups[c].size());
                    std::vector<unsigned int>::const_iterator vit;
                    for(vit = groups[c].begin(); vit != groups[c].end(); ++vit)
                    {
                        cluster_descriptors.push_back(descriptors[*vit]);
                    }

                    meanValue(cluster_descriptors, clusters[c]);
                }

            } // if(!first_time)

            // 2. Associate features with clusters

            // calculate distances to cluster centers
            groups.clear();
            groups.resize(clusters.size(), std::vector<unsigned int>());
            current_association.resize(descriptors.size());

            //assoc.clear();

            //unsigned int d = 0;
            for(auto  fit = descriptors.begin(); fit != descriptors.end(); ++fit)//, ++d)
            {
                double best_dist = distance((*fit), clusters[0]);
                unsigned int icluster = 0;

                for(unsigned int c = 1; c < clusters.size(); ++c)
                {
                    double dist = distance((*fit), clusters[c]);
                    if(dist < best_dist)
                    {
                        best_dist = dist;
                        icluster = c;
                    }
                }

                //assoc.ref<unsigned char>(icluster, d) = 1;

                groups[icluster].push_back(fit - descriptors.begin());
                current_association[ fit - descriptors.begin() ] = icluster;
            }

            // kmeans++ ensures all the clusters has any feature associated with them

            // 3. check convergence
            if(first_time)
            {
                first_time = false;
            }
            else
            {
                //goon = !eqUChar(last_assoc, assoc);

                goon = false;
                for(unsigned int i = 0; i < current_association.size(); i++)
                {
                    if(current_association[i] != last_association[i]){
                        goon = true;
                        break;
                    }
                }
            }

            if(goon)
            {
                // copy last feature-cluster association
                last_association = current_association;
                //last_assoc = assoc.clone();
            }

        } // while(goon)

    } // if must run kmeans

    // create nodes
    for(unsigned int i = 0; i < clusters.size(); ++i)
    {
        NodeId id = m_nodes.size();
        m_nodes.push_back(Node(id));
        m_nodes.back().descriptor = clusters[i];
        m_nodes.back().parent = parent_id;
        m_nodes[parent_id].addChild(id);
    }

    // go on with the next level
    if(current_level < m_L)
    {
        // iterate again with the resulting clusters
        for(unsigned int i = 0; i < m_nodes[parent_id].childNum; ++i)
        {
            NodeId id = m_nodes[parent_id].child[i];

            std::vector<TinyMat> child_features;
            child_features.reserve(groups[i].size());

            std::vector<unsigned int>::const_iterator vit;
            for(vit = groups[i].begin(); vit != groups[i].end(); ++vit)
            {
                child_features.push_back(descriptors[*vit]);
            }

            if(child_features.size() > 1)
            {
                HKmeansStep(id, child_features, current_level + 1);
            }
        }
    }
}

// --------------------------------------------------------------------------


void Vocabulary::initiateClusters
  (const std::vector<TinyMat> &descriptors,
   std::vector<TinyMat> &clusters) const
{
  initiateClustersKMpp(descriptors, clusters);
}

// --------------------------------------------------------------------------


void Vocabulary::initiateClustersKMpp(
  const std::vector<TinyMat> &pfeatures,
    std::vector<TinyMat> &clusters) const
{
  // Implements kmeans++ seeding algorithm
  // Algorithm:
  // 1. Choose one center uniformly at random from among the data points.
  // 2. For each data point x, compute D(x), the distance between x and the nearest
  //    center that has already been chosen.
  // 3. Add one new data point as a center. Each point x is chosen with probability
  //    proportional to D(x)^2.
  // 4. Repeat Steps 2 and 3 until k centers have been chosen.
  // 5. Now that the initial centers have been chosen, proceed using standard k-means
  //    clustering.


//  DUtils::Random::SeedRandOnce();

  clusters.resize(0);
  clusters.reserve(m_k);
  std::vector<double> min_dists(pfeatures.size(), std::numeric_limits<double>::max());

  // 1.

  int ifeature = rand()% pfeatures.size();//DUtils::Random::RandomInt(0, pfeatures.size()-1);

  // create first cluster
  clusters.push_back(pfeatures[ifeature]);

  // compute the initial distances
   std::vector<double>::iterator dit;
  dit = min_dists.begin();
  for(auto fit = pfeatures.begin(); fit != pfeatures.end(); ++fit, ++dit)
  {
    *dit = distance((*fit), clusters.back());
  }

  while((int)clusters.size() < m_k)
  {
    // 2.
    dit = min_dists.begin();
    for(auto  fit = pfeatures.begin(); fit != pfeatures.end(); ++fit, ++dit)
    {
      if(*dit > 0)
      {
        double dist = distance((*fit), clusters.back());
        if(dist < *dit) *dit = dist;
      }
    }

    // 3.
    double dist_sum = std::accumulate(min_dists.begin(), min_dists.end(), 0.0);

    if(dist_sum > 0)
    {
      double cut_d;
      do
      {

        cut_d = (double(rand())/ double(RAND_MAX))* dist_sum;
      } while(cut_d == 0.0);

      double d_up_now = 0;
      for(dit = min_dists.begin(); dit != min_dists.end(); ++dit)
      {
        d_up_now += *dit;
        if(d_up_now >= cut_d) break;
      }

      if(dit == min_dists.end())
        ifeature = pfeatures.size()-1;
      else
        ifeature = dit - min_dists.begin();


      clusters.push_back(pfeatures[ifeature]);
    } // if dist_sum > 0
    else
      break;

  } // while(used_clusters < m_k)

}

// --------------------------------------------------------------------------


void Vocabulary::createWords()
{
  m_words.resize(0);

  if(!m_nodes.empty())
  {
    m_words.reserve( (int)pow((double)m_k, (double)m_L) );


    auto  nit = m_nodes.begin(); // ignore root
    for(++nit; nit != m_nodes.end(); ++nit)
    {
      if(nit->isLeaf())
      {
        nit->word_id = m_words.size();
        m_words.push_back( &(*nit) );
      }
    }
  }
}

// --------------------------------------------------------------------------


void Vocabulary::setNodeWeights
  (const std::vector<std::vector<TinyMat> > &training_features)
{
  const unsigned int NWords = m_words.size();
  const unsigned int NDocs = training_features.size();

  if(m_weighting == TF || m_weighting == BINARY)
  {
    // idf part must be 1 always
    for(unsigned int i = 0; i < NWords; i++)
      m_words[i]->weight = 1;
  }
  else if(m_weighting == IDF || m_weighting == TF_IDF)
  {
    // IDF and TF-IDF: we calculte the idf path now

    // Note: this actually calculates the idf part of the tf-idf score.
    // The complete tf-idf score is calculated in ::transform

    std::vector<unsigned int> Ni(NWords, 0);
    std::vector<bool> counted(NWords, false);


    for(auto mit = training_features.begin(); mit != training_features.end(); ++mit)
    {
      fill(counted.begin(), counted.end(), false);

      for(auto fit = mit->begin(); fit < mit->end(); ++fit)
      {
        WordId word_id;
        transform(*fit, word_id);

        if(!counted[word_id])
        {
          Ni[word_id]++;
          counted[word_id] = true;
        }
      }
    }

    // set ln(N/Ni)
    for(unsigned int i = 0; i < NWords; i++)
    {
      if(Ni[i] > 0)
      {
        m_words[i]->weight = log((double)NDocs / (double)Ni[i]);
      }// else // This cannot occur if using kmeans++
    }

  }

}

// --------------------------------------------------------------------------






// --------------------------------------------------------------------------


float Vocabulary::getEffectiveLevels() const
{
  long sum = 0;
   for(auto wit = m_words.begin(); wit != m_words.end(); ++wit)
  {
    const Node *p = *wit;

    for(; p->id != 0; sum++) p = &m_nodes[p->parent];
  }

  return (float)((double)sum / (double)m_words.size());
}

// --------------------------------------------------------------------------


TinyMat Vocabulary::getWord(WordId wid) const
{
  return m_words[wid]->descriptor;
}

// --------------------------------------------------------------------------


WordValue Vocabulary::getWordWeight(WordId wid) const
{
  return m_words[wid]->weight;
}

// --------------------------------------------------------------------------


WordId Vocabulary::transform
  (const TinyMat& feature) const
{
  if(empty())
  {
    return 0;
  }

  WordId wid;
  transform(feature, wid);
  return wid;
}

// --------------------------------------------------------------------------

void Vocabulary::transform(
        const TinyMat& features, BowVector &v) const
{
    //    std::vector<TinyMat> vf(features.rows);
    //    for(int r=0;r<features.rows;r++) vf[r]=features.rowRange(r,r+1);
    //    transform(vf,v);



    v.clear();

    if(empty())
    {
        return;
    }

    // normalize
    LNorm norm;
    bool must = m_scoring_object->mustNormalize(norm);


    if(m_weighting == TF || m_weighting == TF_IDF)
    {
        for(int r=0;r<features.rows;r++)
        {
            WordId id;
            WordValue w;
            // w is the idf value if TF_IDF, 1 if TF
            transform(features.row(r), id, w);
            // not stopped
            if(w > 0)  addWeight(v,id, w);
        }

        if(!v.empty() && !must)
        {
            // unnecessary when normalizing
            const double nd = v.size();
            for(BowVector::iterator vit = v.begin(); vit != v.end(); vit++)
                vit->second /= nd;
        }

    }
    else // IDF || BINARY
    {
        for(int r=0;r<features.rows;r++)
        {
            WordId id;
            WordValue w;
            // w is idf if IDF, or 1 if BINARY

            transform(features.row(r), id, w);

            // not stopped
            if(w > 0) addIfNotExist(v,id, w);

        } // if add_features
    } // if m_weighting == ...

    if(must) normalize(v,norm);

}



void Vocabulary::transform(const std::vector<TinyMat>& features, BowVector &v) const
{
  v.clear();

  if(empty())
  {
    return;
  }

  // normalize
  LNorm norm;
  bool must = m_scoring_object->mustNormalize(norm);


  if(m_weighting == TF || m_weighting == TF_IDF)
  {
    for(auto fit = features.begin(); fit < features.end(); ++fit)
    {
      WordId id;
      WordValue w;
      // w is the idf value if TF_IDF, 1 if TF

      transform(*fit, id, w);

      // not stopped
      if(w > 0) addWeight(v,id, w);
    }

    if(!v.empty() && !must)
    {
      // unnecessary when normalizing
      const double nd = v.size();
      for(BowVector::iterator vit = v.begin(); vit != v.end(); vit++)
        vit->second /= nd;
    }

  }
  else // IDF || BINARY
  {
    for(auto fit = features.begin(); fit < features.end(); ++fit)
    {
      WordId id;
      WordValue w;
      // w is idf if IDF, or 1 if BINARY

      transform(*fit, id, w);

      // not stopped
      if(w > 0) addIfNotExist(v,id, w);

    } // if add_features
  } // if m_weighting == ...

  if(must) normalize(v,norm);
}

// --------------------------------------------------------------------------


void Vocabulary::transform(
  const std::vector<TinyMat>& features,
  BowVector &v, FeatureVector &fv, int levelsup) const
{
  v.clear();
  fv.clear();

  if(empty()) // safe for subclasses
  {
    return;
  }

  // normalize
  LNorm norm;
  bool must = m_scoring_object->mustNormalize(norm);


  if(m_weighting == TF || m_weighting == TF_IDF)
  {
    unsigned int i_feature = 0;
    for(auto fit = features.begin(); fit < features.end(); ++fit, ++i_feature)
    {
      WordId id;
      NodeId nid;
      WordValue w;
      // w is the idf value if TF_IDF, 1 if TF

      transform(*fit, id, w, &nid, levelsup);

      if(w > 0) // not stopped
      {
        addWeight(v,id, w);
        addFeature(fv,nid, i_feature);
      }
    }

    if(!v.empty() && !must)
    {
      // unnecessary when normalizing
      const double nd = v.size();
      for(BowVector::iterator vit = v.begin(); vit != v.end(); vit++)
        vit->second /= nd;
    }

  }
  else // IDF || BINARY
  {
    unsigned int i_feature = 0;
    for(auto fit = features.begin(); fit < features.end(); ++fit, ++i_feature)
    {
      WordId id;
      NodeId nid;
      WordValue w;
      // w is idf if IDF, or 1 if BINARY

      transform(*fit, id, w, &nid, levelsup);

      if(w > 0) // not stopped
      {
        addIfNotExist(v,id, w);
        addFeature(fv,nid, i_feature);
      }
    }
  } // if m_weighting == ...

  if(must) normalize(v,norm);
}

// --------------------------------------------------------------------------


// --------------------------------------------------------------------------


void Vocabulary::transform
  (const TinyMat &feature, WordId &id) const
{
  WordValue weight;
  transform(feature, id, weight);
}

// --------------------------------------------------------------------------


void Vocabulary::transform(const TinyMat &feature,
  WordId &word_id, WordValue &weight, NodeId *nid, int levelsup) const
{
  // propagate the feature down the tree


  // level at which the node must be stored in nid, if given
  const int nid_level = m_L - levelsup;
  if(nid_level <= 0 && nid != NULL) *nid = 0; // root

  NodeId final_id = 0; // root
  int current_level = 0;

  do
  {
    ++current_level;
    double best_d = std::numeric_limits<double>::max();
//    DescManip::distance(feature, m_nodes[final_id].descriptor);

    for(int i=0;i<m_nodes[final_id].childNum;i++)
    {
        auto id=m_nodes[final_id].child[i];
      double d = distance(feature, m_nodes[id].descriptor);
      if(d < best_d)
      {
        best_d = d;
        final_id = id;
      }
    }

    if(nid != NULL && current_level == nid_level)
      *nid = final_id;

  } while( !m_nodes[final_id].isLeaf() );

  // turn node id into word id
  word_id = m_nodes[final_id].word_id;
  weight = m_nodes[final_id].weight;
}



void Vocabulary::transform(const TinyMat &feature,
  WordId &word_id, WordValue &weight ) const
{
  // propagate the feature down the tree


  // level at which the node must be stored in nid, if given

  NodeId final_id = 0; // root
//maximum speed by computing here distance and avoid calling to DescManip::distance

  //binary descriptor
 // int ntimes=0;
  if (feature.type()==GImageType<uchar,1>::Type){
      do
      {
          const Node& node = m_nodes[final_id];
          uint64_t best_d = std::numeric_limits<uint64_t>::max();
          int idx=0,bestidx=0;
           for(int i=0;i<node.childNum;i++)
          {
               const auto& id=node.child[i];
              //compute distance
             //  std::cout<<idx<< " "<<id<<" "<< m_nodes[id].descriptor<<std::endl;
              uint64_t dist= distance_8uc1(feature, m_nodes[id].descriptor);
              if(dist < best_d)
              {
                  best_d = dist;
                  final_id = id;
                  bestidx=idx;
              }
              idx++;
          }
        // std::cout<<bestidx<<" "<<final_id<<" d:"<<best_d<<" "<<m_nodes[final_id].descriptor<<  std::endl<<std::endl;
      } while( !m_nodes[final_id].isLeaf() );
   }

  // turn node id into word id
  word_id = m_nodes[final_id].word_id;
  weight = m_nodes[final_id].weight;
}
// --------------------------------------------------------------------------

NodeId Vocabulary::getParentNode
  (WordId wid, int levelsup) const
{
  NodeId ret = m_words[wid]->id; // node id
  while(levelsup > 0 && ret != 0) // ret == 0 --> root
  {
    --levelsup;
    ret = m_nodes[ret].parent;
  }
  return ret;
}

// --------------------------------------------------------------------------


void Vocabulary::getWordsFromNode
  (NodeId nid, std::vector<WordId> &words) const
{
  words.clear();

  if(m_nodes[nid].isLeaf())
  {
    words.push_back(m_nodes[nid].word_id);
  }
  else
  {
    words.reserve(m_k); // ^1, ^2, ...

    std::vector<NodeId> parents;
    parents.push_back(nid);

    while(!parents.empty())
    {
      NodeId parentid = parents.back();
      parents.pop_back();

      for(int i=0;i<m_nodes[parentid].childNum;i++)
      {
        const auto id=m_nodes[parentid].child[i];
        const Node &child_node = m_nodes[id];

        if(child_node.isLeaf())
          words.push_back(child_node.word_id);
        else
          parents.push_back(id);

      } // for each child
    } // while !parents.empty
  }
}

// --------------------------------------------------------------------------


int Vocabulary::stopWords(double minWeight)
{
  int c = 0;
   for(auto wit = m_words.begin(); wit != m_words.end(); ++wit)
  {
    if((*wit)->weight < minWeight)
    {
      ++c;
      (*wit)->weight = 0;
    }
  }
  return c;
}

// --------------------------------------------------------------------------


void Vocabulary::save(const std::string &filename,  bool binary_compressed) const
{

}

// --------------------------------------------------------------------------


void Vocabulary::load(const std::string &filename)
{
}
// --------------------------------------------------------------------------

/**
 * Writes printable information of the vocabulary
 * @param os stream to write to
 * @param voc
 */

std::ostream& operator<<(std::ostream &os,
  const Vocabulary &voc)
{
  os << "Vocabulary: k = " << voc.getBranchingFactor()
    << ", L = " << voc.getDepthLevels()
    << ", Weighting = ";

  switch(voc.getWeightingType())
  {
    case Vocabulary::TF_IDF: os << "tf-idf"; break;
    case Vocabulary::TF: os << "tf"; break;
    case Vocabulary::IDF: os << "idf"; break;
    case Vocabulary::BINARY: os << "binary"; break;
  }

  os << ", Scoring = ";
  switch(voc.getScoringType())
  {
    case Vocabulary::L1_NORM: os << "L1-norm"; break;
    case Vocabulary::L2_NORM: os << "L2-norm"; break;
    case Vocabulary::CHI_SQUARE: os << "Chi square distance"; break;
    case Vocabulary::KL: os << "KL-divergence"; break;
    case Vocabulary::BHATTACHARYYA: os << "Bhattacharyya coefficient"; break;
    case Vocabulary::DOT_PRODUCT: os << "Dot product"; break;
  }

  os << ", Number of words = " << voc.size();

  return os;
}
/**
 * @brief Vocabulary::clear
 */
void Vocabulary::clear(){
    m_scoring_object.reset();
    m_scoring_object=0;
    m_nodes.clear();
    m_words.clear();
    m_nodeDescriptors=TinyMat();

}

void Vocabulary::meanValue(const std::vector<TinyMat> &descriptors,
                       TinyMat &mean)
{

    if(descriptors.empty()) return;

    if(descriptors.size() == 1)
    {
        mean = descriptors[0].clone();
        return;
    }
    //binary descriptor
    if (descriptors[0].type()==GImageType<uchar>::Type ){
        //determine number of bytes of the binary descriptor
        int L= descriptors[0].cols*descriptors[0].elemSize();
        std::vector<int> sum( L * 8, 0);

        for(size_t i = 0; i < descriptors.size(); ++i)
        {
            const TinyMat &d = descriptors[i];
            const unsigned char *p = d.data;

            for(int j = 0; j < d.cols; ++j, ++p)
            {
                if(*p & (1 << 7)) ++sum[ j*8     ];
                if(*p & (1 << 6)) ++sum[ j*8 + 1 ];
                if(*p & (1 << 5)) ++sum[ j*8 + 2 ];
                if(*p & (1 << 4)) ++sum[ j*8 + 3 ];
                if(*p & (1 << 3)) ++sum[ j*8 + 4 ];
                if(*p & (1 << 2)) ++sum[ j*8 + 5 ];
                if(*p & (1 << 1)) ++sum[ j*8 + 6 ];
                if(*p & (1))      ++sum[ j*8 + 7 ];
            }
        }

        mean = TinyMat(1, L, GImageType<uchar>::Type);
        memset(mean.data,0,mean.total());
        unsigned char *p = mean.data;

        const int N2 = (int)descriptors.size() / 2 + descriptors.size() % 2;
        for(size_t i = 0; i < sum.size(); ++i)
        {
            if(sum[i] >= N2)
            {
                // set bit
                *p |= 1 << (7 - (i % 8));
            }

            if(i % 8 == 7) ++p;
        }
    }
    //non binary descriptor
    else{
        assert(descriptors[0].type()==GImageType<float>::Type);//ensure it is float

        mean= TinyMat(1, descriptors[0].cols,GImageType<float>::Type);
        memset(mean.data,0,mean.total()*mean.elemSize());
        float inv_s =1./double( descriptors.size());
        for(size_t i=0;i<descriptors.size();i++)
            for(int idx=0;idx<descriptors[i].total();idx++)
            mean.at<float>(idx) +=  ((float*)descriptors[i].data)[idx] * inv_s;

    }
}

double Vocabulary::distance(const TinyMat &a,  const TinyMat &b)
{
    //binary descriptor
    if (a.type()==GImageType<uchar>::Type){

        // Bit count function got from:
         // http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetKernighan
         // This implementation assumes that a.cols (CV_8U) % sizeof(uint64_t) == 0

         const uint64_t *pa, *pb;
         pa = a.ptr<uint64_t>(); // a & b are actually CV_8U
         pb = b.ptr<uint64_t>();

         uint64_t v, ret = 0;
         for(size_t i = 0; i < a.cols / sizeof(uint64_t); ++i, ++pa, ++pb)
         {
           v = *pa ^ *pb;
           v = v - ((v >> 1) & (uint64_t)~(uint64_t)0/3);
           v = (v & (uint64_t)~(uint64_t)0/15*3) + ((v >> 2) &
             (uint64_t)~(uint64_t)0/15*3);
           v = (v + (v >> 4)) & (uint64_t)~(uint64_t)0/255*15;
           ret += (uint64_t)(v * ((uint64_t)~(uint64_t)0/255)) >>
             (sizeof(uint64_t) - 1) * CHAR_BIT;
         }

         return ret;
    }
    else{
        double sqd = 0.;
        assert(a.type()==GImageType<float>::Type);
        assert(a.rows==1);
        const float *a_ptr=a.ptr<float>(0);
        const float *b_ptr=b.ptr<float>(0);
        for(int i = 0; i < a.cols; i ++)
            sqd += (a_ptr[i  ] - b_ptr[i  ])*(a_ptr[i  ] - b_ptr[i  ]);
        return sqd;
    }
}

}


#endif
