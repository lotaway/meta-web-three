package com.metawebthree.action.application;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.action.domain.model.BrandAttention;
import com.metawebthree.action.domain.model.ProductCollection;
import com.metawebthree.action.domain.model.ReadHistory;
import com.metawebthree.action.infrastructure.persistence.mapper.BrandAttentionMapper;
import com.metawebthree.action.infrastructure.persistence.mapper.ProductCollectionMapper;
import com.metawebthree.action.infrastructure.persistence.mapper.ReadHistoryMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;

@Service
@RequiredArgsConstructor
public class UserActionService {
    private final ProductCollectionMapper collectionMapper;
    private final ReadHistoryMapper historyMapper;
    private final BrandAttentionMapper attentionMapper;

    public void addCollection(Long userId, Long productId, String productName, String productPic) {
        ProductCollection collection = ProductCollection.builder()
                .userId(userId)
                .productId(productId)
                .productName(productName)
                .productPic(productPic)
                .createTime(LocalDateTime.now())
                .build();
        collectionMapper.insert(collection);
    }

    public List<ProductCollection> listCollections(Long userId) {
        return collectionMapper.selectList(new LambdaQueryWrapper<ProductCollection>().eq(ProductCollection::getUserId, userId));
    }

    public void deleteCollection(Long userId, Long productId) {
        collectionMapper.delete(new LambdaQueryWrapper<ProductCollection>()
                .eq(ProductCollection::getUserId, userId)
                .eq(ProductCollection::getProductId, productId));
    }

    public void addHistory(Long userId, Long productId, String productName, String productPic) {
        ReadHistory history = ReadHistory.builder()
                .userId(userId)
                .productId(productId)
                .productName(productName)
                .productPic(productPic)
                .createTime(LocalDateTime.now())
                .build();
        historyMapper.insert(history);
    }

    public List<ReadHistory> listHistory(Long userId) {
        return historyMapper.selectList(new LambdaQueryWrapper<ReadHistory>().eq(ReadHistory::getUserId, userId));
    }

    public void addAttention(Long userId, Long brandId, String brandName, String brandLogo) {
        BrandAttention attention = BrandAttention.builder()
                .userId(userId)
                .brandId(brandId)
                .brandName(brandName)
                .brandLogo(brandLogo)
                .createTime(LocalDateTime.now())
                .build();
        attentionMapper.insert(attention);
    }

    public List<BrandAttention> listAttentions(Long userId) {
        return attentionMapper.selectList(new LambdaQueryWrapper<BrandAttention>().eq(BrandAttention::getUserId, userId));
    }
}
