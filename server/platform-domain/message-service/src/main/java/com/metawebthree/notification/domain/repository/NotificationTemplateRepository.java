package com.metawebthree.notification.domain.repository;

import com.metawebthree.notification.domain.model.NotificationTemplateDO;
import java.util.List;

public interface NotificationTemplateRepository {
    NotificationTemplateDO save(NotificationTemplateDO template);
    NotificationTemplateDO findById(Long id);
    NotificationTemplateDO findByCode(String templateCode);
    List<NotificationTemplateDO> findAll();
    boolean deleteById(Long id);
}