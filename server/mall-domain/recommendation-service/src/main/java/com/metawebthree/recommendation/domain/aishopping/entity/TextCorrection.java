package com.metawebthree.recommendation.domain.aishopping.entity;

import java.util.ArrayList;
import java.util.List;

/** Text correction result. */
public class TextCorrection {

    public enum CorrectionSource {
        LLM, LOCAL, NONE
    }

    private String original;
    private String corrected;
    private boolean changed;
    private List<String> suggestions = new ArrayList<>();
    private CorrectionSource source = CorrectionSource.NONE;

    public String getOriginal() {
        return original;
    }

    public void setOriginal(String original) {
        this.original = original;
    }

    public String getCorrected() {
        return corrected;
    }

    public void setCorrected(String corrected) {
        this.corrected = corrected;
    }

    public boolean isChanged() {
        return changed;
    }

    public void setChanged(boolean changed) {
        this.changed = changed;
    }

    public List<String> getSuggestions() {
        return suggestions;
    }

    public void setSuggestions(List<String> suggestions) {
        this.suggestions = suggestions;
    }

    public CorrectionSource getSource() {
        return source;
    }

    public void setSource(CorrectionSource source) {
        this.source = source;
    }
}
