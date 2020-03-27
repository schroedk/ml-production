import logging
import pickle

import pandas as pd
from sensai import InputOutputData
from sensai.data_transformation import DFTNormalisation
from sensai.evaluation import evalModelViaEvaluator
from sensai.featuregen import FeatureGeneratorFromColumnGenerator, FeatureCollector, flattenedFeatureGenerator
from sensai.torch import models

from models.utils import ColumnGeneratorSentenceEncodings, BertBaseMeanEncodingProvider

_log = logging.getLogger(__name__)

if __name__ == "__main__":

    flattenedDf = pd.read_csv("./data/flattenedGifts.csv", index_col="identifier").dropna()

    CACHE_PATH = "sentenceCache.sqlite"
    encodingProvider = BertBaseMeanEncodingProvider()

    def sentenceEmbeddingFeatureGeneratorFactory(persistCache=True):
        columnGen = ColumnGeneratorSentenceEncodings("reviewText", encodingProvider,
            CACHE_PATH, persistCache=persistCache)
        return FeatureGeneratorFromColumnGenerator(columnGen,
            normalisationRuleTemplate=DFTNormalisation.RuleTemplate(unsupported=True))


    def computeFeaturesFromRow(row: pd.Series):
        if not isinstance(row.reviewText, str):
            return
        rowDict = row.asDict()
        rowPandasDf = pd.DataFrame(rowDict, index=[row.identifier])
        print(f"Computing entry for {row.identifier}")
        generator = sentenceEmbeddingFeatureGeneratorFactory()
        generator.generate(rowPandasDf)

    #flattenDf = flattenedDf.apply(computeFeaturesFromRow, axis=1)



    reviewEncodingFeatureGen = sentenceEmbeddingFeatureGeneratorFactory(persistCache=False)
    encodedReviewColName = reviewEncodingFeatureGen.columnGen.generatedColumnName
    flattenedSentenceEncodingsFeatureregen = flattenedFeatureGenerator(reviewEncodingFeatureGen,
        normalisationRuleTemplate=DFTNormalisation.RuleTemplate(skip=True))

    reviewClassifier = models.MultiLayerPerceptronVectorClassificationModel(hiddenDims=[50, 50, 20], cuda=False, epochs=300)
    reviewFeatureCollector = FeatureCollector(flattenedSentenceEncodingsFeatureregen)
    reviewClassifier = reviewClassifier.withFeatureCollector(reviewFeatureCollector)

    targetDf = pd.DataFrame(flattenedDf.pop("overall"))
    inputOutputData = InputOutputData(flattenedDf, targetDf)
    evalModelViaEvaluator(reviewClassifier, inputOutputData, testFraction=0.01, plotTargetDistribution=True)

    with open("reviewClassifier-v1.pickle", 'wb') as f:
        pickle.dump(reviewClassifier, f)

