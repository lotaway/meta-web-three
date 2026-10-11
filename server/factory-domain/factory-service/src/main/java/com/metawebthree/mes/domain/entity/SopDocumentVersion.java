package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;

public class SopDocumentVersion {
    private Long id;
    private Long sopDocumentId;
    private Integer versionNo;
    private String fileName;
    private String filePath;
    private String fileType;
    private Long fileSize;
    private String uploader;
    private String changeDescription;
    private Boolean isCurrentVersion;
    private LocalDateTime uploadedAt;
    private LocalDateTime createdAt;

    public void create(Integer versionNo, String fileName, String filePath, String uploader) {
        this.versionNo = versionNo;
        this.fileName = fileName;
        this.filePath = filePath;
        this.fileType = extractFileType(fileName);
        this.uploader = uploader;
        this.isCurrentVersion = true;
        this.uploadedAt = LocalDateTime.now();
        this.createdAt = LocalDateTime.now();
    }

    private String extractFileType(String fileName) {
        if (fileName == null || !fileName.contains(".")) {
            return "UNKNOWN";
        }
        return fileName.substring(fileName.lastIndexOf(".") + 1).toUpperCase();
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public Long getSopDocumentId() { return sopDocumentId; }
    public void setSopDocumentId(Long sopDocumentId) { this.sopDocumentId = sopDocumentId; }
    public Integer getVersionNo() { return versionNo; }
    public void setVersionNo(Integer versionNo) { this.versionNo = versionNo; }
    public String getFileName() { return fileName; }
    public void setFileName(String fileName) { this.fileName = fileName; }
    public String getFilePath() { return filePath; }
    public void setFilePath(String filePath) { this.filePath = filePath; }
    public String getFileType() { return fileType; }
    public void setFileType(String fileType) { this.fileType = fileType; }
    public Long getFileSize() { return fileSize; }
    public void setFileSize(Long fileSize) { this.fileSize = fileSize; }
    public String getUploader() { return uploader; }
    public void setUploader(String uploader) { this.uploader = uploader; }
    public String getChangeDescription() { return changeDescription; }
    public void setChangeDescription(String changeDescription) { this.changeDescription = changeDescription; }
    public Boolean getIsCurrentVersion() { return isCurrentVersion; }
    public void setIsCurrentVersion(Boolean isCurrentVersion) { this.isCurrentVersion = isCurrentVersion; }
    public LocalDateTime getUploadedAt() { return uploadedAt; }
    public void setUploadedAt(LocalDateTime uploadedAt) { this.uploadedAt = uploadedAt; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
}