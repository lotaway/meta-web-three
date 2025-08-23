package com.metawebthree.common.cloud;

import software.amazon.awssdk.core.sync.RequestBody;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.PutObjectResponse;

import java.io.IOException;

import org.springframework.stereotype.Service;

@Service
public class DefaultS3Service {

    private final S3Client s3Client;

    public DefaultS3Service(S3Client s3Client) {
        this.s3Client = s3Client;
    }

    public S3Client getS3Client() {
        return s3Client;
    }

    public PutObjectResponse putObject(String bucketName, String key, byte[] content) {
        return s3Client.putObject(builder -> builder.bucket(bucketName).key(key), RequestBody.fromBytes(content));
    }

    public PutObjectResponse updateObject(String bucketName, String key, byte[] content) {
        return putObject(bucketName, key, content);
    }

    public byte[] getObject(String bucketName, String key) throws RuntimeException {
        try {
            return s3Client.getObject(builder -> builder.bucket(bucketName).key(key)).readAllBytes();
        } catch (IOException e) {
            throw new RuntimeException("Error while getting object from S3", e);
        }
    }

    public void deleteObject(String bucketName, String key) {
        s3Client.deleteObject(builder -> builder.bucket(bucketName).key(key));
    }
}
