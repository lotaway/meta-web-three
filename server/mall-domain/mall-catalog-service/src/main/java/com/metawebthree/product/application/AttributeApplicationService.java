package com.metawebthree.product.application;

import com.metawebthree.product.domain.model.Attribute;
import com.metawebthree.product.domain.repository.AttributeRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
@RequiredArgsConstructor
public class AttributeApplicationService {

    private final AttributeRepository attributeRepository;

    public void defineAttribute(Attribute attribute) {
        ensureNamePresent(attribute);
        attributeRepository.save(attribute);
    }

    public void modifyAttribute(Attribute attribute) {
        ensureIdPresent(attribute.getId());
        attributeRepository.update(attribute);
    }

    public Attribute getAttribute(Long id) {
        ensureIdPresent(id);
        return attributeRepository.findById(id);
    }

    public List<Attribute> findByCategory(Long categoryId) {
        return attributeRepository.findByCategoryId(categoryId);
    }

    public void removeAttribute(Long id) {
        ensureIdPresent(id);
        attributeRepository.delete(id);
    }

    private void ensureNamePresent(Attribute attr) {
        if (attr.getName() == null || attr.getName().isEmpty()) {
            throw new IllegalArgumentException("Attribute name is required");
        }
    }

    private void ensureIdPresent(Long id) {
        if (id == null) {
            throw new IllegalArgumentException("Attribute id must be valid");
        }
    }
}
