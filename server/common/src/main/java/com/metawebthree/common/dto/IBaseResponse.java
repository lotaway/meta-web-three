package com.metawebthree.common.dto;

import com.metawebthree.common.enums.ResponseStatus;

public class IBaseResponse<Data> {
    ResponseStatus status;
    String message;
    Data data;
}
