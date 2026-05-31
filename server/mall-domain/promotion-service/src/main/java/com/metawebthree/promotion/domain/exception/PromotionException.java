package com.metawebthree.promotion.domain.exception;

import com.metawebthree.common.exception.BusinessException;
import com.metawebthree.common.enums.ResponseStatus;

public class PromotionException extends BusinessException {
    private final PromotionErrorCode errorCode;

    public PromotionException(PromotionErrorCode errorCode, String message) {
        super(ResponseStatus.SYSTEM_ERROR, message);
        this.errorCode = errorCode;
    }

    public PromotionErrorCode getErrorCode() {
        return errorCode;
    }
}
