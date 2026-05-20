package com.metawebthree.product.application;

import com.metawebthree.product.domain.model.Brand;
import com.metawebthree.product.domain.repository.BrandRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
@RequiredArgsConstructor
public class BrandApplicationService {

    private final BrandRepository brandRepository;

    public void registerBrand(Brand brand) {
        validateBrand(brand);
        brandRepository.save(brand);
    }

    public void modifyBrand(Brand brand) {
        validateId(brand.getId());
        brandRepository.update(brand);
    }

    public Brand getBrand(Long id) {
        validateId(id);
        return brandRepository.findById(id);
    }

    public List<Brand> listBrands() {
        return brandRepository.findAllBySort();
    }

    public void removeBrand(Long id) {
        validateId(id);
        brandRepository.delete(id);
    }

    private void validateBrand(Brand brand) {
        if (brand.getName() == null || brand.getName().isEmpty()) {
            throw new IllegalArgumentException("Brand name is required");
        }
    }

    private void validateId(Long id) {
        if (id == null) {
            throw new IllegalArgumentException("Brand identity must be provided");
        }
    }
}
