package com.metawebthree.cs.domain.model.enums;

/**
 * Customer sentiment/emotion categories for AI customer service analysis.
 */
public enum Sentiment {
    /**
     * Positive sentiment - customer is satisfied, happy, grateful
     */
    POSITIVE,
    
    /**
     * Neutral sentiment - factual inquiry, no emotional charge
     */
    NEUTRAL,
    
    /**
     * Negative sentiment - slightly dissatisfied, annoyed
     */
    NEGATIVE,
    
    /**
     * Very negative sentiment - angry, frustrated, demanding escalation
     */
    ANGRY,
    
    /**
     * Anxious/worried sentiment - seeking reassurance
     */
    ANXIOUS,
    
    /**
     * Confused sentiment - needs clarification
     */
    CONFUSED
}