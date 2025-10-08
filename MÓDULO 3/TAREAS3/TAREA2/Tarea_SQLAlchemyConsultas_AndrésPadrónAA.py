#!/usr/bin/env python
# coding: utf-8

# # <font color='blue'>Tarea Consultas en SQL</font>
# ### <font color='red'>13 de octubre de 2025</font>
# ### Andrés Padrón Quintana 
# ### <font color='green'>Curso : DATA SCIENCE AND MACHINE LEARNING APPLIED TO FINANCIAL MARKETS </font>

# In[1]:


#Librerías a usar
import pandas as pd
from sqlalchemy import create_engine

# Crear conexión
engine = create_engine("mysql+pymysql://root:Fergateodio7@localhost:3306/banco_base")


# ## Consultas a realizar (Bloque 1-Fácil)
# 

# 1. Lista los nombres y edades de todos los clientes que tengan ingresos mayores a
# 50,000.

# In[4]:


query1 = """
SELECT nombre, edad
FROM clientes
WHERE ingresos > 50000;
"""
pd.read_sql(query1, engine)


# 2. Obtén el número total de clientes por país, ordenado de mayor a menor.

# In[5]:


query2 = """
SELECT pais, COUNT(*) AS total_clientes
FROM clientes
GROUP BY pais
ORDER BY total_clientes DESC;
"""
pd.read_sql(query2, engine)


# 3. Muestra los clientes (id_cliente, nombre, ingresos) cuyo estrato sea 2 y que
# además tengan al menos una tarjeta asociada.

# In[11]:


pd.read_sql("SHOW TABLES;", engine)


# In[12]:


query3 = """
SELECT c.id_cliente, c.nombre, c.ingresos
FROM clientes c
JOIN tarjetas t ON c.id_cliente = t.cliente
WHERE c.estrato = 2;
"""
pd.read_sql(query3, engine)


# 4. Calcula el promedio de ingresos por sexo, y muestra solo aquellos cuyo promedio
# sea mayor a 40,000.

# In[13]:


query4 = """
SELECT sexo, AVG(ingresos) AS promedio_ingresos
FROM clientes
GROUP BY sexo
HAVING AVG(ingresos) > 40000;
"""
pd.read_sql(query4, engine)


# 5. Encuentra los 5 clientes con mayores ingresos y ordénalos de manera
# descendente.

# In[14]:


query5 = """
SELECT id_cliente, nombre, ingresos
FROM clientes
ORDER BY ingresos DESC
LIMIT 5;
"""
pd.read_sql(query5, engine)


# 6. Realiza un JOIN para mostrar el nombre del cliente, el tipo de tarjeta y el monto
# asociado.

# In[15]:


query6 = """
SELECT c.nombre, t.tipo, CAST(t.monto AS DECIMAL(15,2)) AS monto
FROM clientes c
JOIN tarjetas t ON c.id_cliente = t.cliente;
"""
pd.read_sql(query6, engine)


# 7. Encuentra a los clientes que tienen más de una tarjeta registrada.

# In[16]:


query7 = """
SELECT c.id_cliente, c.nombre, COUNT(t.id_tarjeta) AS num_tarjetas
FROM clientes c
JOIN tarjetas t ON c.id_cliente = t.cliente
GROUP BY c.id_cliente, c.nombre
HAVING COUNT(t.id_tarjeta) > 1;
"""
pd.read_sql(query7, engine)


# 8. Obtén el nombre y país de los clientes cuyo ingreso sea mayor al ingreso promedio
# de todos los clientes. (Usa una subconsulta).

# In[17]:


query8 = """
SELECT nombre, pais, ingresos
FROM clientes
WHERE ingresos > (SELECT AVG(ingresos) FROM clientes);
"""
pd.read_sql(query8, engine)


# 9. Usa una función de ventana para mostrar el nombre de cada cliente, su país y el
# ranking de ingresos dentro de su país.

# In[18]:


query9 = """
SELECT nombre, pais, ingresos,
       RANK() OVER (PARTITION BY pais ORDER BY ingresos DESC) AS ranking
FROM clientes;
"""
pd.read_sql(query9, engine)


# 10. Encuentra los clientes que poseen tarjeta de crédito y cuyos ingresos estén en el
# top 10% de todos los ingresos. (Usa percent_rank o ntile).

# In[19]:


query10 = """
WITH ingresos_rank AS (
    SELECT c.id_cliente, c.nombre, c.ingresos, t.tipo,
           PERCENT_RANK() OVER (ORDER BY c.ingresos) AS pr
    FROM clientes c
    JOIN tarjetas t ON c.id_cliente = t.cliente
    WHERE t.tipo = 'Credito'
)
SELECT id_cliente, nombre, ingresos, tipo, pr
FROM ingresos_rank
WHERE pr >= 0.9;
"""
pd.read_sql(query10, engine)


# ## Consultas a realizar (Bloque 2-Intermedio)

# 1. Top 3 países con mayor promedio de ingresos entre clientes con tarjetas de crédito.
# (Pista: join + filtro por tipo de tarjeta + group by + order by con limit).

# In[22]:


query11 = """
SELECT c.pais, AVG(c.ingresos) AS promedio_ingresos
FROM clientes c
JOIN tarjetas t ON c.id_cliente = t.cliente
WHERE t.tipo = 'Credito'
GROUP BY c.pais
ORDER BY promedio_ingresos DESC
LIMIT 3;
"""
pd.read_sql(query11, engine)


# 2. Obtener el nombre y país de los clientes cuyo ingreso es mayor al ingreso
# promedio de su mismo estrato.
# (Pista: subconsulta correlacionada por estrato).

# In[23]:


query12 = """
SELECT c.nombre, c.pais, c.ingresos, c.estrato
FROM clientes c
WHERE c.ingresos > (
    SELECT AVG(c2.ingresos)
    FROM clientes c2
    WHERE c2.estrato = c.estrato
);
"""
pd.read_sql(query12, engine)


# 3. Mostrar el cliente con más tarjetas en cada país.
# (Pista: join + group by + función de ventana RANK() OVER (PARTITION BY pais
# ORDER BY COUNT(*) DESC)).

# In[24]:


query13 = """
SELECT id_cliente, nombre, pais, num_tarjetas
FROM (
    SELECT c.id_cliente, c.nombre, c.pais,
           COUNT(t.id_tarjeta) AS num_tarjetas,
           RANK() OVER (PARTITION BY c.pais ORDER BY COUNT(t.id_tarjeta) DESC) AS rnk
    FROM clientes c
    JOIN tarjetas t ON c.id_cliente = t.cliente
    GROUP BY c.id_cliente, c.nombre, c.pais
) sub
WHERE rnk = 1;
"""
pd.read_sql(query13, engine)


# 4. Calcular el monto total de tarjetas por cliente y mostrar solo aquellos cuya
# suma de montos supere el promedio de todos los clientes.
# (Pista: agregación con having + subconsulta del promedio).

# In[25]:


query14 = """
SELECT c.id_cliente, c.nombre, SUM(CAST(t.monto AS DECIMAL(15,2))) AS total_monto
FROM clientes c
JOIN tarjetas t ON c.id_cliente = t.cliente
GROUP BY c.id_cliente, c.nombre
HAVING SUM(CAST(t.monto AS DECIMAL(15,2))) > (
    SELECT AVG(total) 
    FROM (
        SELECT SUM(CAST(monto AS DECIMAL(15,2))) AS total
        FROM tarjetas
        GROUP BY cliente
    ) x
);
"""
pd.read_sql(query14, engine)


# 5. Obtener los 5 clientes más jóvenes que tienen tarjeta de crédito y cuyos
# ingresos estén en el top 10% de todos los ingresos.
# (Pista: percentil con NTILE(10) OVER (ORDER BY ingresos DESC) y filtro).

# In[26]:


query15 = """
WITH ingresos_deciles AS (
    SELECT c.id_cliente, c.nombre, c.edad, c.ingresos, t.tipo,
           NTILE(10) OVER (ORDER BY c.ingresos DESC) AS decil
    FROM clientes c
    JOIN tarjetas t ON c.id_cliente = t.cliente
    WHERE t.tipo = 'Credito'
)
SELECT id_cliente, nombre, edad, ingresos
FROM ingresos_deciles
WHERE decil = 1
ORDER BY edad ASC
LIMIT 5;
"""
pd.read_sql(query15, engine)


# 6. Mostrar los clientes que tienen tanto tarjeta de débito como de crédito.
# (Pista: self-join o group by con having count distinct).

# In[27]:


query16 = """
SELECT c.id_cliente, c.nombre
FROM clientes c
JOIN tarjetas t ON c.id_cliente = t.cliente
GROUP BY c.id_cliente, c.nombre
HAVING COUNT(DISTINCT t.tipo) = 2;
"""
pd.read_sql(query16, engine)


# 7. Calcular el ingreso promedio por estrato y sexo, y mostrar solo las
# combinaciones donde ese promedio sea mayor al promedio general de
# ingresos.
# (Pista: doble group by con having y subconsulta).

# In[28]:


query17 = """
SELECT estrato, sexo, AVG(ingresos) AS promedio_ingresos
FROM clientes
GROUP BY estrato, sexo
HAVING AVG(ingresos) > (SELECT AVG(ingresos) FROM clientes);
"""
pd.read_sql(query17, engine)


# 8. Obtener para cada país el cliente con mayor monto total en tarjetas.
# (Pista: join + sum(monto) + función ROW_NUMBER() OVER (PARTITION BY pais
# ORDER BY SUM(monto) DESC)).

# In[29]:


query18 = """
SELECT id_cliente, nombre, pais, total_monto
FROM (
    SELECT c.id_cliente, c.nombre, c.pais,
           SUM(CAST(t.monto AS DECIMAL(15,2))) AS total_monto,
           ROW_NUMBER() OVER (PARTITION BY c.pais ORDER BY SUM(CAST(t.monto AS DECIMAL(15,2))) DESC) AS rn
    FROM clientes c
    JOIN tarjetas t ON c.id_cliente = t.cliente
    GROUP BY c.id_cliente, c.nombre, c.pais
) sub
WHERE rn = 1;
"""
pd.read_sql(query18, engine)


# 9. Listar a los clientes cuyo ingreso está por encima del promedio de su país, y
# además ese país tenga al menos 5 clientes registrados.
# (Pista: subconsulta + having).

# In[30]:


query19 = """
SELECT c.id_cliente, c.nombre, c.pais, c.ingresos
FROM clientes c
WHERE c.ingresos > (
    SELECT AVG(c2.ingresos)
    FROM clientes c2
    WHERE c2.pais = c.pais
)
AND c.pais IN (
    SELECT pais
    FROM clientes
    GROUP BY pais
    HAVING COUNT(*) >= 5
);
"""
pd.read_sql(query19, engine)


# 10. Calcular el monto promedio de tarjetas de crédito por estrato y mostrar en
# qué estrato es más alto.
# (Pista: agregación + order by + limit 1).

# In[31]:


query20 = """
SELECT estrato, AVG(CAST(t.monto AS DECIMAL(15,2))) AS promedio_credito
FROM clientes c
JOIN tarjetas t ON c.id_cliente = t.cliente
WHERE t.tipo = 'Credito'
GROUP BY estrato
ORDER BY promedio_credito DESC
LIMIT 1;
"""
pd.read_sql(query20, engine)

