package com.metawebthree.promotion.application;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

import com.metawebthree.promotion.domain.enums.CouponStatus;
import com.metawebthree.promotion.domain.model.Coupon;
import com.metawebthree.promotion.domain.model.CouponType;
import com.metawebthree.promotion.domain.ports.CouponRepository;
import com.metawebthree.promotion.domain.ports.CouponTypeRepository;
import com.metawebthree.promotion.domain.ports.TimeProvider;

public class CouponQueryServiceTest {

    @Test
    public void validateShouldReturnPayable() {
        TimeProvider timeProvider = () -> LocalDateTime.of(2024, 1, 1, 10, 0);
        InMemoryCouponTypeRepository typeRepository = new InMemoryCouponTypeRepository();
        InMemoryCouponRepository couponRepository = new InMemoryCouponRepository();

        CouponType type = new CouponType();
        type.setId(1L);
        type.setName("A");
        type.setMinimumOrderAmount(new BigDecimal("100"));
        type.setDiscountAmount(new BigDecimal("10"));
        type.setIsEnabled(true);
        type.setStartTime(LocalDateTime.of(2023, 1, 1, 0, 0));
        type.setEndTime(LocalDateTime.of(2025, 1, 1, 0, 0));
        typeRepository.save(type);

        Coupon coupon = new Coupon();
        coupon.setCode("CODE1");
        coupon.setCouponTypeId(1L);
        coupon.setOwnerUserId(9L);
        coupon.setUseStatus(CouponStatus.UNUSED.getCode());
        couponRepository.save(coupon);

        CouponQueryService service = new CouponQueryService(couponRepository, typeRepository, timeProvider);
        CouponQueryService.CouponValidateResult result = service.validate("CODE1", 9L,
                new BigDecimal("120"), new BigDecimal("5"));

        assertNotNull(result);
        assertEquals(new BigDecimal("10"), result.getDiscountAmount());
        assertEquals(new BigDecimal("115"), result.getPayableAmount());
    }

    static class InMemoryCouponTypeRepository implements CouponTypeRepository {
        private final Map<Long, CouponType> store = new HashMap<>();

        @Override
        public void save(CouponType couponType) {
            store.put(couponType.getId(), couponType);
        }

        @Override
        public CouponType findById(Long id) {
            return store.get(id);
        }
    }

    static class InMemoryCouponRepository implements CouponRepository {
        private final Map<String, Coupon> store = new HashMap<>();

        @Override
        public void save(Coupon coupon) {
            store.put(coupon.getCode(), coupon);
        }

        @Override
        public void saveAll(List<Coupon> coupons) {
            for (Coupon coupon : coupons) {
                save(coupon);
            }
        }

        @Override
        public Coupon findByCode(String code) {
            return store.get(code);
        }

        @Override
        public Coupon findFirstAvailableByType(Long couponTypeId) {
            return null;
        }

        @Override
        public List<Coupon> findAvailableByType(Long couponTypeId, int limit) {
            return new ArrayList<>();
        }

        @Override
        public void updateOwnerIfAvailable(Long couponId, Long ownerUserId, Integer acquireMethod) {
        }

        @Override
        public void updateOwnerForCode(String code, Long ownerUserId, Integer acquireMethod, Integer transferStatus) {
        }

        @Override
        public void updateStatusToUsed(String code, Long ownerUserId, String orderNo, String consumerName, String operatorName) {
        }

        @Override
        public void updateTransferStatus(String code, Long ownerUserId, Integer transferStatus) {
        }

        @Override
        public List<Coupon> listByOwner(Long ownerUserId, Integer useStatus) {
            return new ArrayList<>();
        }

        @Override
        public List<Coupon> listByBatch(String batchId) {
            return new ArrayList<>();
        }
    }
}
