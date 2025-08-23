package com.metawebthree.common.adapter;

import java.sql.Array;
import java.sql.CallableStatement;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Arrays;
import java.util.stream.Collectors;

import org.apache.ibatis.type.BaseTypeHandler;
import org.apache.ibatis.type.JdbcType;
import org.postgresql.util.PGobject;

public class IntegerArrayTypeHandler extends BaseTypeHandler<Integer[]> {

    @Override
    public void setNonNullParameter(PreparedStatement ps, int i, Integer[] parameter, JdbcType jdbcType) throws SQLException {
        // PGobject pgObject = new PGobject();
        // pgObject.setType("integer[]");
        // pgObject.setValue(Arrays.stream(parameter).map(String::valueOf).collect(Collectors.joining(",")));
        // ps.setObject(i, pgObject);
        Connection conn = ps.getConnection();
        Array array = conn.createArrayOf("integer", parameter);
        ps.setArray(i, array);
    }

    @Override
    public Integer[] getNullableResult(ResultSet rs, String columnName) throws SQLException {
        return getResultFromArray(rs.getArray(columnName));
    }

    @Override
    public Integer[] getNullableResult(ResultSet rs, int columnIndex) throws SQLException {
        return getResultFromArray(rs.getArray(columnIndex));
    }

    @Override
    public Integer[] getNullableResult(CallableStatement cs, int columnIndex) throws SQLException {
        return getResultFromArray(cs.getArray(columnIndex));
    }

    private Integer[] getResultFromArray(java.sql.Array array) throws SQLException {
        if (array == null) {
            return null;
        }
        Integer[] result = (Integer[]) array.getArray();
        return result;
    }
}