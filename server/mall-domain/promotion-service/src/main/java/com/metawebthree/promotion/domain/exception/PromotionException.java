package com.metawebthree.promotion.domain.exception;

public class PromotionException extends RuntimeException {
    private final PromotionErrorCode errorCode;

    public PromotionException(PromotionErrorCode errorCode, String message) {
        super(message);
        this.errorCode = errorCode;
    }

    public PromotionErrorCode getErrorCode() {
        return errorCode;
    }
}
