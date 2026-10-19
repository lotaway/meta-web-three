package com.metawebthree.rma.application.integration;

/**
 * Outcome of a stock-in call to the inventory service.
 */
public class StockInResult {

    private final boolean success;
    private final boolean duplicated;
    private final String message;

    private StockInResult(boolean success, boolean duplicated, String message) {
        this.success = success;
        this.duplicated = duplicated;
        this.message = message;
    }

    public static StockInResult success(boolean duplicated) {
        return new StockInResult(true, duplicated, duplicated ? "Already stocked in" : "Stock-in successful");
    }

    public static StockInResult failure(String message) {
        return new StockInResult(false, false, message);
    }

    public boolean isSuccess() {
        return success;
    }

    public boolean isDuplicated() {
        return duplicated;
    }

    public String getMessage() {
        return message;
    }
}
