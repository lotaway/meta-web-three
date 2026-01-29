package com.metawebthree.media;

import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.common.cloud.DefaultS3Config;
import com.metawebthree.common.cloud.DefaultS3Service;
import com.metawebthree.media.DO.ArtWorkDO;

import lombok.RequiredArgsConstructor;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import software.amazon.awssdk.services.s3.model.PutObjectResponse;

import java.io.IOException;
import java.util.Optional;
import java.util.Set;

@Slf4j
@Service
@RequiredArgsConstructor
public class MediaService {
    private static final Set<String> ALLOWED_SIZES = Set.of("64x64", "128x128", "256x256", "512x512", "1024x1024");
    private final DefaultS3Service s3Service;

    private final DefaultS3Config s3Config;

    private final ArtWorkMapper artWorkMapper;

    public Boolean createMediaMetadata(ArtWorkDO artWorkDO) {
        artWorkMapper.insert(artWorkDO);
        return true;
    }

    public PutObjectResponse createMedia(String key, byte[] content) {
        return s3Service.putObject(s3Config.getName(), key, content);
    }

    public PutObjectResponse updateMedia(String key, byte[] content) {
        return s3Service.putObject(s3Config.getName(), key, content);
    }
    
    public Optional<byte[]> getMedia(String key) {
        // Handle dynamic resizing: key@WxH
        if (key.contains("@")) {
            String[] parts = key.split("@");
            String originalKey = parts[0];
            String sizeSuffix = parts[1]; // e.g., 128x128

            if (!ALLOWED_SIZES.contains(sizeSuffix)) {
                log.warn("Requested image size not allowed: {}", sizeSuffix);
                return s3Service.getObject(s3Config.getName(), originalKey);
            }

            // 1. Try to get the already resized version from S3
            Optional<byte[]> cached = s3Service.getObject(s3Config.getName(), key);
            if (cached.isPresent()) {
                return cached;
            }

            // 2. Fetch original and resize
            Optional<byte[]> original = s3Service.getObject(s3Config.getName(), originalKey);
            if (original.isPresent()) {
                try {
                    String[] dims = sizeSuffix.split("x");
                    int width = Integer.parseInt(dims[0]);
                    int height = Integer.parseInt(dims[1]);

                    java.io.ByteArrayOutputStream os = new java.io.ByteArrayOutputStream();
                    net.coobird.thumbnailator.Thumbnails.of(new java.io.ByteArrayInputStream(original.get()))
                            .size(width, height)
                            .outputFormat("jpg") // Default output format
                            .toOutputStream(os);
                    
                    byte[] resizedData = os.toByteArray();
                    
                    // 3. Cache the resized version back to S3 for future use
                    s3Service.putObject(s3Config.getName(), key, resizedData);
                    
                    return Optional.of(resizedData);
                } catch (Exception e) {
                    log.error("Failed to resize image: {}", key, e);
                    return original; // Fallback to original
                }
            }
            return Optional.empty();
        }
        
        return s3Service.getObject(s3Config.getName(), key);
    }
    
    public String getFileUrl(String key) {
        return s3Service.getFileUrlWithCheck(s3Config.getName(), key).orElse(null);
    }

    public void deleteMedia(String key) {
        s3Service.deleteObject(s3Config.getName(), key);
    }

    public String uploadFile(MultipartFile file) {
        String key = String.valueOf(IdWorker.getId());
        try {
            s3Service.putObject(s3Config.getName(), key, file.getBytes());
            return key;
        } catch (IOException e) {
            throw new RuntimeException("Failed to upload file", e);
        }
    }
}
