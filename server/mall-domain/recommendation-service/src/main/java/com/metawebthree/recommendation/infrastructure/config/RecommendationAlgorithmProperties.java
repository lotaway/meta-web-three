package com.metawebthree.recommendation.infrastructure.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "recommendation.algorithm")
public class RecommendationAlgorithmProperties {

    private Scoring scoring = new Scoring();
    private CTR ctr = new CTR();
    private Conversion conversion = new Conversion();

    public Scoring getScoring() { return scoring; }
    public void setScoring(Scoring scoring) { this.scoring = scoring; }
    public CTR getCtr() { return ctr; }
    public void setCtr(CTR ctr) { this.ctr = ctr; }
    public Conversion getConversion() { return conversion; }
    public void setConversion(Conversion conversion) { this.conversion = conversion; }

    public static class Scoring {
        private double collaborativeWeight = 0.4;
        private double contentWeight = 0.3;
        private double hybridWeight = 0.5;
        private double popularityWeight = 0.6;
        private double aiModelWeight = 0.8;
        private double baseScore = 80.0;
        private double scoreDecay = 5.0;

        public double getCollaborativeWeight() { return collaborativeWeight; }
        public void setCollaborativeWeight(double collaborativeWeight) { this.collaborativeWeight = collaborativeWeight; }
        public double getContentWeight() { return contentWeight; }
        public void setContentWeight(double contentWeight) { this.contentWeight = contentWeight; }
        public double getHybridWeight() { return hybridWeight; }
        public void setHybridWeight(double hybridWeight) { this.hybridWeight = hybridWeight; }
        public double getPopularityWeight() { return popularityWeight; }
        public void setPopularityWeight(double popularityWeight) { this.popularityWeight = popularityWeight; }
        public double getAiModelWeight() { return aiModelWeight; }
        public void setAiModelWeight(double aiModelWeight) { this.aiModelWeight = aiModelWeight; }
        public double getBaseScore() { return baseScore; }
        public void setBaseScore(double baseScore) { this.baseScore = baseScore; }
        public double getScoreDecay() { return scoreDecay; }
        public void setScoreDecay(double scoreDecay) { this.scoreDecay = scoreDecay; }
    }

    public static class CTR {
        private double industryAverage = 3.5;
        private double highThreshold = 5.0;
        private double lowThreshold = 1.0;

        public double getIndustryAverage() { return industryAverage; }
        public void setIndustryAverage(double industryAverage) { this.industryAverage = industryAverage; }
        public double getHighThreshold() { return highThreshold; }
        public void setHighThreshold(double highThreshold) { this.highThreshold = highThreshold; }
        public double getLowThreshold() { return lowThreshold; }
        public void setLowThreshold(double lowThreshold) { this.lowThreshold = lowThreshold; }
    }

    public static class Conversion {
        private double industryAverage = 1.2;
        private double highThreshold = 2.5;
        private double lowThreshold = 0.5;

        public double getIndustryAverage() { return industryAverage; }
        public void setIndustryAverage(double industryAverage) { this.industryAverage = industryAverage; }
        public double getHighThreshold() { return highThreshold; }
        public void setHighThreshold(double highThreshold) { this.highThreshold = highThreshold; }
        public double getLowThreshold() { return lowThreshold; }
        public void setLowThreshold(double lowThreshold) { this.lowThreshold = lowThreshold; }
    }
}