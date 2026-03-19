package com.metawebthree.promotion.application;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

import com.metawebthree.promotion.domain.model.Coupon;
import com.metawebthree.promotion.domain.model.CouponBatch;
import com.metawebthree.promotion.domain.model.CouponType;
import com.metawebthree.promotion.domain.ports.CodeGenerator;
import com.metawebthree.promotion.domain.ports.CouponBatchRepository;
import com.metawebthree.promotion.domain.ports.CouponRepository;
import com.metawebthree.promotion.domain.ports.CouponTypeRepository;
import com.metawebthree.promotion.domain.ports.TimeProvider;

public class CouponCommandServiceTest {

    @Test
    public void generateBatchCreatesCoupons() {
        TimeProvider timeProvider = () -> LocalDateTime.of(2024, 1, 1, 10, 0);
        InMemoryCouponRepository couponRepository = new InMemoryCouponRepository();
        InMemoryCouponTypeRepository couponTypeRepository = new InMemoryCouponTypeRepository();
        InMemoryCouponBatchRepository batchRepository = new InMemoryCouponBatchRepository();
        CodeGenerator codeGenerator = new SequenceCodeGenerator();
        CouponPolicy policy = new CouponPolicy(3, 5);

        CouponType type = new CouponType();
        type.setId(1L);
        type.setIsEnabled(true);
        type.setStartTime(LocalDateTime.of(2023, 1, 1, 0, 0));
        type.setEndTime(LocalDateTime.of(2025, 1, 1, 0, 0));
        couponTypeRepository.save(type);

        CouponCommandService service = new CouponCommandService(couponRepository, couponTypeRepository,
                batchRepository, codeGenerator, timeProvider, policy);
        service.generateBatch("B1", 1L, 2);

        assertEquals(2, couponRepository.items.size());
        assertEquals(1, batchRepository.items.size());
    }

    static class SequenceCodeGenerator implements CodeGenerator {
        private int value;
        @Override
        public String nextCode() {
            value += 1;
            return "CODE" + value;
        }
    }

    static class InMemoryCouponRepository implements CouponRepository {
        private final Map<String, Coupon> items = new HashMap<>();

        @Override
        public void save(Coupon coupon) {
            items.put(coupon.getCode(), coupon);
        }

        @Override
        public void saveAll(List<Coupon> coupons) {
            for (Coupon coupon : coupons) {
                save(coupon);
            }
        }

        @Override
        public Coupon findByCode(String code) {
            return items.get(code);
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
            return new ArrayList<>(items.values());
        }
    }

    static class InMemoryCouponTypeRepository implements CouponTypeRepository {
        private final Map<Long, CouponType> items = new HashMap<>();

        @Override
        public void save(CouponType couponType) {
            items.put(couponType.getId(), couponType);
        }

        @Override
        public CouponType findById(Long id) {
            return items.get(id);
        }
    }

    static class InMemoryCouponBatchRepository implements CouponBatchRepository {
        private final Map<String, CouponBatch> items = new HashMap<>();

        @Override
        public void save(CouponBatch batch) {
            items.put(batch.getId(), batch);
        }

        @Override
        public CouponBatch findById(String id) {
            return items.get(id);
        }
    }
}
