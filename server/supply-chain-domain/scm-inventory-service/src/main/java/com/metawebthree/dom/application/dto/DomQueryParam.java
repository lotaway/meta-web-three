package com.metawebthree.dom.application.dto;

import lombok.Data;

@Data
public class DomQueryParam {
    private String status;
    private String domOrderNo;
    private Integer pageNum = 1;
    private Integer pageSize = 20;
}
