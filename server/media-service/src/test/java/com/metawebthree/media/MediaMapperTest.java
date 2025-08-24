package com.metawebthree.media;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.ibatis.cursor.Cursor;
import org.apache.ibatis.executor.BatchResult;
import org.apache.ibatis.session.ExecutorType;
import org.apache.ibatis.session.SqlSession;
import org.apache.ibatis.session.SqlSessionFactory;
import org.junit.Assert;
import org.junit.Test;
import org.junit.jupiter.api.Disabled;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.test.context.junit4.SpringRunner;
import org.springframework.transaction.annotation.Transactional;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.baomidou.mybatisplus.core.toolkit.LambdaUtils;
import com.baomidou.mybatisplus.extension.toolkit.SqlHelper;
import com.github.yulichang.wrapper.MPJLambdaWrapper;
import com.metawebthree.media.DO.ArtWorkDO;
import com.metawebthree.media.DO.PeopleDO;
import com.metawebthree.media.DO.PeopleTypeDO;

@RunWith(SpringRunner.class)
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.DEFINED_PORT)
@ComponentScan(basePackages = {
        "com.metawebthree.media",
        "com.metawebthree.common"
})
@EnableAutoConfiguration
public class MediaMapperTest {

    @Autowired
    private ArtWorkMapper artWorkMapper;

    @Autowired
    private PeopleMapper peopleMapper;

    @Autowired
    private PeopleTypeMapper peopleTypeMapper;

    @Test
    @Disabled("Fix data")
    public void fixArtWorkDirector() {
        String directorName = "文舟";
        String type = "director";

        PeopleTypeDO targetPeopleTypeDO = PeopleTypeDO.builder().type(type).build();
        PeopleTypeDO matchPeopleTypeDO = getOrCreatePeopleType(targetPeopleTypeDO);

        PeopleDO targetPeopleDO = PeopleDO.builder().name(directorName).types(new Short[] { matchPeopleTypeDO.getId() })
                .build();
        PeopleDO matchPeopleDO = getOrCreatePeople(targetPeopleDO);

        UpdateWrapper<ArtWorkDO> updateWrapper = new UpdateWrapper<ArtWorkDO>().eq("series", "骑士的沙丘").set("director",
                matchPeopleDO.getId());
        int result = artWorkMapper.update(updateWrapper);
        Assert.assertTrue(result > 0);
    }

    public PeopleTypeDO getOrCreatePeopleType(PeopleTypeDO targetPeopleTypeDO) {
        PeopleTypeDO matchPeopleTypeDO = peopleTypeMapper.selectOne(new MPJLambdaWrapper<PeopleTypeDO>()
                .select(PeopleTypeDO::getId).eq(PeopleTypeDO::getType, targetPeopleTypeDO.getType()));
        if (matchPeopleTypeDO.equals(null)) {
            int getTypeResult = peopleTypeMapper.insert(targetPeopleTypeDO);
            Assert.assertEquals(1, getTypeResult);
        }
        return matchPeopleTypeDO;
    }

    public PeopleDO getOrCreatePeople(PeopleDO targetPeopleDO) {
        MPJLambdaWrapper<PeopleDO> queryWrapper = new MPJLambdaWrapper<>();
        queryWrapper.select(PeopleDO::getId).eq(PeopleDO::getName, targetPeopleDO.getName());
        PeopleDO matchPeopleDO = peopleMapper.selectOne(queryWrapper);
        if (matchPeopleDO.equals(null)) {
            int result = peopleMapper.insert(matchPeopleDO);
            Assert.assertEquals(1, result);
        }
        return matchPeopleDO;
    }

    @Test
    public void fixSubtitleUndefinedError() {
        String errorPart = "undefined";
        ArtWorkDO artWorkDO = ArtWorkDO.builder().subtitle(errorPart).build();
        UpdateWrapper<ArtWorkDO> updateWrapper = new UpdateWrapper<>();
        String fieldName = "subtitle";
        updateWrapper.likeRight(fieldName, artWorkDO.getSubtitle())
                .setSql(String.format("%s = REPLACE(%s, '%s', '')", fieldName, errorPart, fieldName));
        artWorkMapper.update(updateWrapper);
        ArtWorkDO errorColumn = artWorkMapper.selectOne(
                new MPJLambdaWrapper<ArtWorkDO>()
                        .select(ArtWorkDO::getId)
                        .like(ArtWorkDO::getSubtitle, "undefined"));
        Assert.assertNull(errorColumn);
    }

    @Test
    public void fixArtworkSeriesError() {
        List<Integer> results = processArtworks();
        Assert.assertTrue(results.size() > 0);
        Assert.assertTrue(results.stream().allMatch(i -> i > 0));
    }

    @Transactional
    public List<Integer> processArtworks() {
        List<Integer> results = new ArrayList<>();
        try (Cursor<ArtWorkDO> cursor = artWorkMapper.getCursor()) {
            List<ArtWorkDO> buffer = new ArrayList<>();
            int batchSize = 1000;
            for (ArtWorkDO data : cursor) {
                data = processArtwork(data);
                buffer.add(data);
                if (buffer.size() >= batchSize) {
                    results.addAll(batchUpdate(buffer));
                    buffer.clear();
                }
            }
            if (!buffer.isEmpty()) {
                results.addAll(batchUpdate(buffer));
                buffer.clear();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return results;
    }

    private List<Integer> batchUpdate(List<ArtWorkDO> list) {
        List<Integer> results = new ArrayList<>();
        SqlSessionFactory factory = SqlHelper.sqlSessionFactory(ArtWorkDO.class);
        try (SqlSession session = factory.openSession(ExecutorType.BATCH)) {
            ArtWorkMapper mapper = session.getMapper(ArtWorkMapper.class);
            for (ArtWorkDO u : list) {
                results.add(mapper.updateById(u));
            }
            session.flushStatements();
        }
        return results;
    }

    private ArtWorkDO processArtwork(ArtWorkDO artWorkDO) {
        if (artWorkDO.getTitle().matches("\\d{1,2}$")) {
            artWorkDO.setTitle(artWorkDO.getTitle());
        }
        return artWorkDO;
    }
}
