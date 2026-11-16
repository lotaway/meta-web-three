package com.metawebthree.common.autoconfigure;

import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Import;
import org.springframework.web.multipart.MultipartFile;

import com.metawebthree.common.cloud.DefaultS3Config;
import com.metawebthree.common.cloud.DefaultS3Service;
import com.metawebthree.common.cloud.LocalStorageService;
import com.metawebthree.common.cloud.MinioConfig;
import com.metawebthree.common.cloud.MinioService;
import com.metawebthree.common.cloud.MinioStorageService;
import com.metawebthree.common.cloud.S3StorageService;

/**
 * Object storage beans (local / MinIO / S3), opt-in per service. The concrete
 * implementation is still selected by {@code storage.type}.
 */
@AutoConfiguration
@ConditionalOnClass(MultipartFile.class)
@ConditionalOnProperty(prefix = "meta.common.storage.object", name = "enabled", havingValue = "true")
@EnableConfigurationProperties(CommonStorageProperties.class)
@Import({
        MinioConfig.class,
        MinioService.class,
        MinioStorageService.class,
        DefaultS3Config.class,
        DefaultS3Service.class,
        S3StorageService.class,
        LocalStorageService.class
})
public class ObjectStorageAutoConfiguration {
}
