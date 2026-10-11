package com.metawebthree.promotion.infrastructure.persistence.mapper;

import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Mapper
public interface FlashOrderMapper {

    @Insert("INSERT INTO tb_order (id, member_id, order_sn, order_type, order_status, total_amount, pay_amount, promotion_amount, promotion_info, order_remark, receiver_name, receiver_phone, receiver_province, receiver_city, receiver_region, receiver_detail_address, member_receive_address_id, delete_status, create_time, updated_at) "
            + "VALUES (#{id}, #{userId}, #{orderSn}, 1, 'CREATED', #{totalAmount}, #{payAmount}, #{promotionAmount}, '闪购', #{remark}, #{receiverName}, #{receiverPhone}, #{receiverProvince}, #{receiverCity}, #{receiverRegion}, #{receiverDetailAddress}, #{receiveAddressId}, 0, #{now}, #{now})")
    int insertOrder(@Param("id") Long id, @Param("userId") Long userId, @Param("orderSn") String orderSn,
                    @Param("totalAmount") BigDecimal totalAmount, @Param("payAmount") BigDecimal payAmount,
                    @Param("promotionAmount") BigDecimal promotionAmount,
                    @Param("remark") String remark,
                    @Param("receiverName") String receiverName, @Param("receiverPhone") String receiverPhone,
                    @Param("receiverProvince") String receiverProvince, @Param("receiverCity") String receiverCity,
                    @Param("receiverRegion") String receiverRegion, @Param("receiverDetailAddress") String receiverDetailAddress,
                    @Param("receiveAddressId") Long receiveAddressId,
                    @Param("now") LocalDateTime now);
}
