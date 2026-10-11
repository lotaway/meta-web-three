package com.metawebthree.logistics.infrastructure.persistence.repository;

import com.metawebthree.logistics.domain.entity.Carrier;
import java.util.List;
import java.util.Optional;

public interface CarrierRepository {

    Optional<Carrier> findById(Long id);

    Optional<Carrier> findByCarrierCode(String carrierCode);

    List<Carrier> findAll();

    Carrier save(Carrier carrier);

    void delete(Carrier carrier);
}