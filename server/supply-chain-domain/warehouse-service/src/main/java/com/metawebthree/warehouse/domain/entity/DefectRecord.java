package com.metawebthree.warehouse.domain.entity;

import lombok.Data;
import java.time.LocalDateTime;

/**
 * 不良品记录实体
 * 记录每个不良品项
 */
@Data
public class DefectRecord {
    private Long id;
    private Long inspectionId;
    private Long inspectionItemId;
    private String skuCode;
    private String productName;
    private String batchNo;
    private String defectType;
    private String defectName;
    private String defectDescription;
    private Integer defectQuantity;
    private String defectLevel;
    private String photoUrls;
    private String locationCode;
    private String creator;
    private LocalDateTime createTime;
    private String updater;
    private LocalDateTime updateTime;
    private Boolean deleted;

    public static final String TYPE_APPEARANCE = "外观缺陷";
    public static final String TYPE_QUANTITY_SHORT = "数量短缺";
    public static final String TYPE_PACKAGE_DAMAGE = "包装破损";
    public static final String TYPE_LABEL_ERROR = "标签错误";
    public static final String TYPE_SPEC_MISMATCH = "规格不符";
    public static final String TYPE_EXPIRED = "过期";
    public static final String TYPE_OTHER = "其他";

    public static final String LEVEL_CRITICAL = "CRITICAL";
    public static final String LEVEL_MAJOR = "MAJOR";
    public static final String LEVEL_MINOR = "MINOR";

    public boolean isCritical() {
        return LEVEL_CRITICAL.equals(this.defectLevel);
    }

    public boolean isMajor() {
        return LEVEL_MAJOR.equals(this.defectLevel);
    }

    public boolean isMinor() {
        return LEVEL_MINOR.equals(this.defectLevel);
    }
}