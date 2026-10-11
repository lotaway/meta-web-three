package com.metawebthree.digitaltwin.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.digitaltwin.domain.entity.ProductionLine;
import com.metawebthree.digitaltwin.domain.repository.ProductionLineRepository;
import com.metawebthree.digitaltwin.infrastructure.persistence.converter.ProductionLineConverter;
import com.metawebthree.digitaltwin.infrastructure.persistence.dataobject.ProductionLineDO;
import com.metawebthree.digitaltwin.infrastructure.persistence.mapper.ProductionLineMapper;
import org.springframework.stereotype.Repository;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class ProductionLineRepositoryImpl implements ProductionLineRepository {

    private final ProductionLineMapper productionLineMapper;

    public ProductionLineRepositoryImpl(ProductionLineMapper productionLineMapper) {
        this.productionLineMapper = productionLineMapper;
    }

    @Override
    public Optional<ProductionLine> findById(Long id) {
        return Optional.ofNullable(productionLineMapper.selectById(id))
                .map(ProductionLineConverter::toEntity);
    }

    @Override
    public Optional<ProductionLine> findByLineCode(String lineCode) {
        ProductionLineDO d = productionLineMapper.selectOne(
                new LambdaQueryWrapper<ProductionLineDO>().eq(ProductionLineDO::getLineCode, lineCode));
        return Optional.ofNullable(ProductionLineConverter.toEntity(d));
    }

    @Override
    public List<ProductionLine> findByWorkshopId(String workshopId) {
        return productionLineMapper.selectList(
                new LambdaQueryWrapper<ProductionLineDO>().eq(ProductionLineDO::getWorkshopId, workshopId))
                .stream().map(ProductionLineConverter::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<ProductionLine> findByStatus(ProductionLine.ProductionLineStatus status) {
        return productionLineMapper.selectList(
                new LambdaQueryWrapper<ProductionLineDO>().eq(ProductionLineDO::getStatus, status.name()))
                .stream().map(ProductionLineConverter::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<ProductionLine> findAll() {
        return productionLineMapper.selectList(null)
                .stream().map(ProductionLineConverter::toEntity).collect(Collectors.toList());
    }

    @Override
    public IPage<ProductionLine> findPaginated(int page, int size) {
        Page<ProductionLineDO> pageObj = new Page<>(page, size);
        IPage<ProductionLineDO> result = productionLineMapper.selectPage(pageObj, null);
        return result.convert(ProductionLineConverter::toEntity);
    }

    @Override
    public ProductionLine save(ProductionLine line) {
        ProductionLineDO d = ProductionLineConverter.toDO(line);
        productionLineMapper.insert(d);
        line.setId(d.getId());
        return line;
    }

    @Override
    public void update(ProductionLine line) {
        ProductionLineDO d = ProductionLineConverter.toDO(line);
        productionLineMapper.updateById(d);
    }

    @Override
    public void deleteById(Long id) {
        productionLineMapper.deleteById(id);
    }
}
