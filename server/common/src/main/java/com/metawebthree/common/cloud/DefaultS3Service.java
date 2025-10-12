package com.metawebthree.common.cloud;

import software.amazon.awssdk.core.sync.RequestBody;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.ObjectCannedACL;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;
import software.amazon.awssdk.services.s3.model.PutObjectResponse;
import software.amazon.awssdk.services.s3.model.S3Exception;

import java.io.IOException;
import java.util.Optional;

import org.apache.dubbo.remoting.http12.HttpStatus;
import org.springframework.stereotype.Service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@Service
@RequiredArgsConstructor
public class DefaultS3Service {

    private final S3Client s3Client;

    public S3Client getS3Client() {
        return s3Client;
    }

    public PutObjectResponse putObject(String bucketName, String key, byte[] content) {
        return s3Client.putObject(builder -> builder.bucket(bucketName).key(key), RequestBody.fromBytes(content));
    }

    public PutObjectResponse updateObject(String bucketName, String key, byte[] content) {
        return putObject(bucketName, key, content);
    }

    public Optional<byte[]> getObject(String bucketName, String key) {
        try {
            return Optional.of(s3Client.getObject(builder -> builder.bucket(bucketName).key(key)).readAllBytes());
        } catch (S3Exception e) {
            if (e.statusCode() == HttpStatus.NOT_FOUND.getCode()) {
                return Optional.empty();
            } else {
                throw e;
            }
        } catch (IOException e) {
            throw new RuntimeException("Error while getting object from S3", e);
        }
    }

    public void deleteObject(String bucketName, String key) {
        s3Client.deleteObject(builder -> builder.bucket(bucketName).key(key));
    }

    public String uploadExcel(String bucketName, byte[] excelBytes, String fileName) {
        return uploadExcel(bucketName, excelBytes, fileName, false);
    }

    public String uploadExcel(String bucketName, byte[] excelBytes, String fileName, boolean isPublic) {
        try {
            RequestBody requestBody = RequestBody.fromBytes(excelBytes);
            PutObjectRequest putObjectRequest = PutObjectRequest.builder()
                    .bucket(bucketName)
                    .key(fileName)
                    .contentType("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    .acl(isPublic ? ObjectCannedACL.PUBLIC_READ : ObjectCannedACL.PRIVATE)
                    .build();
            PutObjectResponse putObjectResponse = s3Client.putObject(putObjectRequest, requestBody);
            putObjectResponse.bucketKeyEnabled();
            String fileUrl = getFileUrl(bucketName, fileName);
            log.info("Excel file uploaded to S3: {}", fileUrl);
            return fileUrl;
        } catch (Exception e) {
            log.error("Failed to upload Excel file to S3", e);
            throw new RuntimeException("Failed to upload Excel file to S3", e);
        }
    }

    public String getFileUrl(String bucketName, String fileName) {
        return String.format("https://%s.s3.amazonaws.com/%s", bucketName, fileName);
    }

    public Optional<String> getFileUrlWithCheck(String bucketName, String fileName) {
        if (getObject(bucketName, fileName).isPresent()) {
            return Optional.of(String.format("https://%s.s3.amazonaws.com/%s", bucketName, fileName));
        }
        return Optional.empty();
    }
}
