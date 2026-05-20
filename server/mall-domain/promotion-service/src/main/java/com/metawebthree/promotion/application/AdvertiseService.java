package com.metawebthree.promotion.application;

import com.metawebthree.promotion.domain.model.Advertise;
import com.metawebthree.promotion.domain.ports.AdvertiseRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
@RequiredArgsConstructor
public class AdvertiseService {
    private final AdvertiseRepository advertiseRepository;

    public void create(Advertise advertise) {
        advertiseRepository.save(advertise);
    }

    public void update(Advertise advertise) {
        advertiseRepository.update(advertise);
    }

    public void delete(Long id) {
        advertiseRepository.delete(id);
    }

    public Advertise getById(Long id) {
        return advertiseRepository.findById(id);
    }

    public List<Advertise> list(String name, Integer type, String endTime, Integer status) {
        return advertiseRepository.list(name, type, endTime, status);
    }

    public List<Advertise> listAvailable(Integer type) {
        return advertiseRepository.listAvailable(type);
    }
}
