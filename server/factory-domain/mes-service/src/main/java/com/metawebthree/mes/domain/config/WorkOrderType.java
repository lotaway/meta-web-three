package com.metawebthree.mes.domain.config;

import lombok.Data;
import jakarta.validation.constraints.*;

@Data
public class WorkOrderType {
    @NotBlank(message = "typeCode不能为空")
    @Size(max = 50, message = "typeCode长度不能超过50")
    private String typeCode;
    
    @NotBlank(message = "typeName不能为空")
    @Size(max = 100, message = "typeName长度不能超过100")
    private String typeName;
    
    @Size(max = 500, message = "description长度不能超过500")
    private String description;
    
    @Size(max = 50, message = "statusMachineCode长度不能超过50")
    private String statusMachineCode;
    
    @Size(max = 50, message = "processRouteTemplate长度不能超过50")
    private String processRouteTemplate;
    
    private Boolean isDefault;
    
    @Min(value = 0, message = "sortOrder不能为负数")
    @Max(value = 999, message = "sortOrder不能超过999")
    private Integer sortOrder;
    
    @Size(max = 20, message = "status长度不能超过20")
    private String status;
    
    private Long id;
    
    public static final String TYPE_NORMAL = "NORMAL";
    public static final String TYPE_REWORK = "REWORK";
    public static final String TYPE_REPAIR = "REPAIR";
    public static final String TYPE_SAMPLE = "SAMPLE";
}