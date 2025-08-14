package com.metawebthree.common.dto;

import com.metawebthree.common.enums.ResponseStatus;

public class IBaseResponse<D> {
    ResponseStatus status;
    String message;
    D data;
}
