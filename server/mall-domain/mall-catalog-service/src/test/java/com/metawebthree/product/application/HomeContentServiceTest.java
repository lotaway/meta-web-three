package com.metawebthree.product.application;

import com.baomidou.mybatisplus.core.conditions.Wrapper;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.product.domain.model.BrandDO;
import com.metawebthree.product.domain.model.PrefrenceAreaDO;
import com.metawebthree.product.domain.model.ProductCategoryDO;
import com.metawebthree.product.domain.model.SubjectDO;
import com.metawebthree.product.dto.HomeContentDTO;
import com.metawebthree.product.dto.ProductDTO;
import com.metawebthree.product.infrastructure.persistence.mapper.BrandMapper;
import com.metawebthree.product.infrastructure.persistence.mapper.PrefrenceAreaMapper;
import com.metawebthree.product.infrastructure.persistence.mapper.ProductCategoryMapper;
import com.metawebthree.product.infrastructure.persistence.mapper.SubjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Homepage content must come from persisted operational records, not hard-coded
 * mock banners.
 */
class HomeContentServiceTest {

    private ProductService productService;
    private ProductCategoryMapper productCategoryMapper;
    private SubjectMapper subjectMapper;
    private PrefrenceAreaMapper prefrenceAreaMapper;
    private BrandMapper brandMapper;
    private HomeContentService service;

    @BeforeEach
    void setUp() {
        productService = mock(ProductService.class);
        productCategoryMapper = mock(ProductCategoryMapper.class);
        subjectMapper = mock(SubjectMapper.class);
        prefrenceAreaMapper = mock(PrefrenceAreaMapper.class);
        brandMapper = mock(BrandMapper.class);
        service = new HomeContentService(
                productService, productCategoryMapper, subjectMapper, prefrenceAreaMapper, brandMapper);
    }

    private SubjectDO subject(Long id, String title, Long categoryId) {
        SubjectDO subject = new SubjectDO();
        subject.setId(id);
        subject.setTitle(title);
        subject.setPic("https://cdn.example.com/subject-" + id + ".jpg");
        subject.setCategoryId(categoryId);
        subject.setShowStatus(1);
        return subject;
    }

    private Page<SubjectDO> subjectPage(List<SubjectDO> records) {
        Page<SubjectDO> page = new Page<>(1, 10);
        page.setRecords(records);
        page.setTotal(records.size());
        return page;
    }

    @Test
    void subjectListReadsPersistedSubjects() {
        when(subjectMapper.selectPage(any(Page.class), any(Wrapper.class)))
                .thenReturn(subjectPage(List.of(
                        subject(1L, "夏季清仓", 7L),
                        subject(2L, "新品首发", 8L))));

        List<HomeContentDTO.SubjectDTO> result = service.subjectList(null, 1, 10);

        assertEquals(2, result.size());
        assertEquals("夏季清仓", result.get(0).getTitle());
        assertEquals(7L, result.get(0).getCategoryId());
        assertFalse(result.get(0).getPic().contains("placeholder"));
    }

    @Test
    void subjectListFiltersByCategoryAndShowStatus() {
        when(subjectMapper.selectPage(any(Page.class), any(Wrapper.class)))
                .thenReturn(subjectPage(List.of()));

        service.subjectList(7L, 1, 10);

        ArgumentCaptor<QueryWrapper<SubjectDO>> captor = ArgumentCaptor.forClass(QueryWrapper.class);
        verify(subjectMapper).selectPage(any(Page.class), captor.capture());
        String segment = captor.getValue().getSqlSegment();
        assertTrue(segment.contains("show_status"), segment);
        assertTrue(segment.contains("category_id"), segment);
    }

    @Test
    void productCateListReadsCategoriesFromDatabase() {
        ProductCategoryDO category = new ProductCategoryDO();
        category.setId(3L);
        category.setParentId(0L);
        category.setName("家电");
        category.setLevel(1);
        category.setIcon("home.png");
        when(productCategoryMapper.selectList(any(Wrapper.class))).thenReturn(List.of(category));

        List<HomeContentDTO.CategoryDTO> result = service.productCateList(0L);

        assertEquals(1, result.size());
        assertEquals("家电", result.get(0).getName());
        assertEquals(0L, result.get(0).getParentId());
        assertEquals("home.png", result.get(0).getIcon());
    }

    @Test
    void advertiseListIsDrivenByPreferenceAreas() {
        PrefrenceAreaDO area = new PrefrenceAreaDO();
        area.setId(5L);
        area.setName("限时秒杀");
        area.setPic("https://cdn.example.com/sale.jpg");
        area.setShowStatus(1);
        when(prefrenceAreaMapper.selectList(any(Wrapper.class))).thenReturn(List.of(area));

        List<HomeContentDTO.AdvertiseDTO> result = service.listPreferenceAreas();

        assertEquals(1, result.size());
        assertEquals("限时秒杀", result.get(0).getName());
        assertEquals("https://cdn.example.com/sale.jpg", result.get(0).getPic());
    }

    @Test
    void contentAssemblesProductsSubjectsAndBannersFromDatabase() {
        ProductDTO product = new ProductDTO();
        product.setId(1);
        product.setName("新品");
        when(productService.listProducts(any(), any(), any())).thenReturn(List.of(product));
        when(subjectMapper.selectPage(any(Page.class), any(Wrapper.class)))
                .thenReturn(subjectPage(List.of(subject(1L, "专题", null))));
        BrandDO brand = new BrandDO();
        brand.setId(9L);
        brand.setName("品牌A");
        when(brandMapper.selectList(any(Wrapper.class))).thenReturn(List.of(brand));
        PrefrenceAreaDO area = new PrefrenceAreaDO();
        area.setId(5L);
        area.setName("运营位");
        when(prefrenceAreaMapper.selectList(any(Wrapper.class))).thenReturn(List.of(area));

        HomeContentDTO content = service.content();

        assertEquals(1, content.getNewProductList().size());
        assertEquals(1, content.getHotProductList().size());
        assertEquals(1, content.getBrandList().size());
        assertEquals("品牌A", content.getBrandList().get(0).getName());
        assertEquals(1, content.getSubjectList().size());
        assertEquals("运营位", content.getAdvertiseList().get(0).getName());
    }

    @Test
    void recommendProductListDelegatesWithPagination() {
        when(productService.listProducts(any(), any(), any(), any(), any())).thenReturn(List.of());

        service.recommendProductList(2, 15);

        verify(productService).listProducts(null, null, null, 2, 15);
    }
}
