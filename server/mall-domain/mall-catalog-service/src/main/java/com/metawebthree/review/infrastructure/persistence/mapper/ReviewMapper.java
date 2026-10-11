package com.metawebthree.review.infrastructure.persistence.mapper;

import com.metawebthree.review.domain.model.ReviewDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import java.util.List;

@Mapper
public interface ReviewMapper {
    int insert(ReviewDO record);
    int update(ReviewDO record);
    int deleteById(Long id);
    ReviewDO selectById(Long id);
    List<ReviewDO> selectByProductId(Long productId);
    List<ReviewDO> selectByUserId(Long userId);
    List<ReviewDO> selectByStoreId(Long storeId);
    List<ReviewDO> selectByOrderId(Long orderId);
    List<ReviewDO> selectAll();
    int updateStatus(@Param("id") Long id, @Param("status") Integer status);
    int incrementLikeCount(@Param("id") Long id);
    int addReply(@Param("id") Long id, @Param("replyContent") String replyContent);
}