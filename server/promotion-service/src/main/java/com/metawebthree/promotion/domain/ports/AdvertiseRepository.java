package com.metawebthree.promotion.domain.ports;

import com.metawebthree.promotion.domain.model.Advertise;
import java.util.List;

public interface AdvertiseRepository {
    void save(Advertise advertise);
    void update(Advertise advertise);
    void delete(Long id);
    Advertise findById(Long id);
    List<Advertise> list(String name, Integer type, String endTime, Integer status);
    List<Advertise> listAvailable(Integer type);
}
