package com.metawebthree.cs.application;

import com.metawebthree.cs.domain.model.enums.Sentiment;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Service;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * Sentiment analysis for customer service messages.
 * Uses keyword-based analysis to detect customer emotion.
 */
@Service
public class SentimentAnalysisService {
    private static final Logger log = LoggerFactory.getLogger(SentimentAnalysisService.class);

    private static final Map<Sentiment, List<Pattern>> POSITIVE_PATTERNS = new HashMap<>();
    private static final Map<Sentiment, List<Pattern>> NEGATIVE_PATTERNS = new HashMap<>();
    private static final Map<Sentiment, List<Pattern>> ANXIOUS_PATTERNS = new HashMap<>();
    private static final Map<Sentiment, List<Pattern>> CONFUSED_PATTERNS = new HashMap<>();

    static {
        // Positive patterns
        POSITIVE_PATTERNS.put(Sentiment.POSITIVE, Arrays.asList(
                Pattern.compile("thank|thanks|appreciate|grateful|excellent|great|wonderful|amazing", Pattern.CASE_INSENSITIVE),
                Pattern.compile("完美|很好|感谢|谢谢|满意", Pattern.CASE_INSENSITIVE),
                Pattern.compile("\\bi love it\\b|\\bi like\\b|\\bgood job\\b|\\bwell done\\b", Pattern.CASE_INSENSITIVE)
        ));

        // Negative (frustrated) patterns
        NEGATIVE_PATTERNS.put(Sentiment.NEGATIVE, Arrays.asList(
                Pattern.compile("bad|terrible|awful|disappointed|frustrat", Pattern.CASE_INSENSITIVE),
                Pattern.compile("失望|糟糕|太差|不满", Pattern.CASE_INSENSITIVE),
                Pattern.compile("\\bthis is useless\\b|\\bnot helpful\\b|\\bwaste of time\\b", Pattern.CASE_INSENSITIVE),
                Pattern.compile("why.*always|never.*works|still.*not working", Pattern.CASE_INSENSITIVE)
        ));

        // Angry patterns
        NEGATIVE_PATTERNS.put(Sentiment.ANGRY, Arrays.asList(
                Pattern.compile("angry|mad|furious|horrible|hate|worst|complain", Pattern.CASE_INSENSITIVE),
                Pattern.compile("讨厌|愤怒|恶心|差评|投诉|曝光", Pattern.CASE_INSENSITIVE),
                Pattern.compile("\\bspeak to.*manager\\b|\\bwant.*refund.*now\\b|\\bcancel.*immediately\\b", Pattern.CASE_INSENSITIVE),
                Pattern.compile("\\bfucking?|\\bdamn|\\bshit", Pattern.CASE_INSENSITIVE)
        ));

        // Anxious patterns
        ANXIOUS_PATTERNS.put(Sentiment.ANXIOUS, Arrays.asList(
                Pattern.compile("worri|concern|afraid|uncertain|nervous|panic", Pattern.CASE_INSENSITIVE),
                Pattern.compile("担心|着急|焦虑|紧张|怎么办|会不会|靠谱", Pattern.CASE_INSENSITIVE),
                Pattern.compile("\\bwhen will|\\bhow long.*take|\\bis it safe\\b", Pattern.CASE_INSENSITIVE),
                Pattern.compile("\\bmy money|\\bmypackage|\\bLost|\\bmissing", Pattern.CASE_INSENSITIVE)
        ));

        // Confused patterns
        CONFUSED_PATTERNS.put(Sentiment.CONFUSED, Arrays.asList(
                Pattern.compile("confused|dont understand|unclear|how do|what.*mean", Pattern.CASE_INSENSITIVE),
                Pattern.compile("不懂|不清楚|怎么|什么是|如何操作", Pattern.CASE_INSENSITIVE),
                Pattern.compile("\\bwhere is|\\bwhich one|\\bwhich option", Pattern.CASE_INSENSITIVE),
                Pattern.compile("\\bhelp me|\\bplease guide|\\bexplain", Pattern.CASE_INSENSITIVE)
        ));
    }

    /**
     * Analyze sentiment of a customer message.
     *
     * @param message the customer message content
     * @return detected sentiment
     */
    @Cacheable(value = "sentiment", key = "#message.hashCode()")
    public Sentiment analyze(String message) {
        if (message == null || message.trim().isEmpty()) {
            return Sentiment.NEUTRAL;
        }

        try {
            int angryScore = matchPatterns(message, NEGATIVE_PATTERNS.get(Sentiment.ANGRY));
            if (angryScore >= 2) {
                return Sentiment.ANGRY;
            }

            int negativeScore = matchPatterns(message, NEGATIVE_PATTERNS.get(Sentiment.NEGATIVE));
            if (negativeScore >= 2) {
                return Sentiment.NEGATIVE;
            }

            int positiveScore = matchPatterns(message, POSITIVE_PATTERNS.get(Sentiment.POSITIVE));
            if (positiveScore >= 2) {
                return Sentiment.POSITIVE;
            }

            int anxiousScore = matchPatterns(message, ANXIOUS_PATTERNS.get(Sentiment.ANXIOUS));
            if (anxiousScore >= 2) {
                return Sentiment.ANXIOUS;
            }

            int confusedScore = matchPatterns(message, CONFUSED_PATTERNS.get(Sentiment.CONFUSED));
            if (confusedScore >= 2) {
                return Sentiment.CONFUSED;
            }

            // Check softer signals
            if (positiveScore >= 1 || anxiousScore >= 1 || confusedScore >= 1) {
                if (positiveScore >= anxiousScore && positiveScore >= confusedScore) {
                    return Sentiment.POSITIVE;
                }
                if (anxiousScore >= 1) {
                    return Sentiment.ANXIOUS;
                }
                if (confusedScore >= 1) {
                    return Sentiment.CONFUSED;
                }
            }

            return Sentiment.NEUTRAL;
        } catch (Exception e) {
            log.warn("Sentiment analysis failed: {}", e.getMessage());
            return Sentiment.NEUTRAL;
        }
    }

    private int matchPatterns(String message, List<Pattern> patterns) {
        if (patterns == null) {
            return 0;
        }
        int score = 0;
        for (Pattern pattern : patterns) {
            if (pattern.matcher(message).find()) {
                score++;
            }
        }
        return score;
    }

    /**
     * Batch analyze multiple messages.
     *
     * @param messages list of messages to analyze
     * @return map of message to sentiment
     */
    public Map<String, Sentiment> analyzeBatch(List<String> messages) {
        Map<String, Sentiment> results = new HashMap<>();
        for (String message : messages) {
            results.put(message, analyze(message));
        }
        return results;
    }
}