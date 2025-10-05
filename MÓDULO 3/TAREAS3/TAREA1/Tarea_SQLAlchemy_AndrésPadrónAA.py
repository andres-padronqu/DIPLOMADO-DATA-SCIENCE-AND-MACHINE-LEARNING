#!/usr/bin/env python
# coding: utf-8

# # <font color='blue'>Tarea SQLAlchemy</font>
# ### <font color='red'>13 de octubre de 2025</font>
# ### Andrés Padrón Quintana 
# ### <font color='green'>Curso : DATA SCIENCE AND MACHINE LEARNING APPLIED TO FINANCIAL MARKETS </font>

# In[1]:


# -------------------------
# Importar funciones y declarar la base de datos
# -------------------------

from sqlalchemy import (
    create_engine, Column, Integer, String, Date, ForeignKey, UniqueConstraint, CheckConstraint
)
from sqlalchemy.orm import declarative_base, relationship, validates
import re

Base = declarative_base()


# In[2]:


# -------------------------
# Tabla para Estudiante
# -------------------------
class Estudiante(Base):
    __tablename__ = 'estudiantes'

    id = Column(Integer, primary_key=True, autoincrement=True)
    nombre = Column(String(100), nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    matricula = Column(String(8), unique=True, nullable=False)
    fecha_inscripcion = Column(Date, nullable=False)

    inscripciones = relationship("Inscripcion", back_populates="estudiante")

    # Validaciones
    @validates("email")
    def validate_email(self, key, email):
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            raise ValueError("Email inválido")
        return email

    @validates("matricula")
    def validate_matricula(self, key, matricula):
        if len(matricula) != 8 or not matricula.isalnum():
            raise ValueError("La matrícula debe tener exactamente 8 caracteres alfanuméricos")
        return matricula


# In[3]:


# -------------------------
# Tabla para Profesor
# -------------------------
class Profesor(Base):
    __tablename__ = 'profesores'

    id = Column(Integer, primary_key=True, autoincrement=True)
    nombre = Column(String(100), nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    departamento = Column(String(50), nullable=False)

    cursos = relationship("Curso", back_populates="profesor")

    @validates("email")
    def validate_email(self, key, email):
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            raise ValueError("Email inválido")
        return email


# In[4]:


# -------------------------
# Tabla para Curso
# -------------------------
class Curso(Base):
    __tablename__ = 'cursos'

    id = Column(Integer, primary_key=True, autoincrement=True)
    nombre = Column(String(80), nullable=False)
    creditos = Column(Integer, nullable=False)
    nivel = Column(String(20), nullable=False)
    profesor_id = Column(Integer, ForeignKey("profesores.id"), nullable=False)

    profesor = relationship("Profesor", back_populates="cursos")
    inscripciones = relationship("Inscripcion", back_populates="curso")

    __table_args__ = (
        CheckConstraint("creditos >= 1 AND creditos <= 10", name="chk_creditos"),
        CheckConstraint("nivel IN ('Licenciatura','Maestría','Doctorado')", name="chk_nivel"),
    )


# In[5]:


# -------------------------
# Tabla Inscripcion (N-M)
# -------------------------
class Inscripcion(Base):
    __tablename__ = 'inscripciones'

    id = Column(Integer, primary_key=True, autoincrement=True)
    estudiante_id = Column(Integer, ForeignKey("estudiantes.id"), nullable=False)
    curso_id = Column(Integer, ForeignKey("cursos.id"), nullable=False)
    fecha_inscripcion = Column(Date, nullable=False)
    calificacion = Column(Integer, nullable=True)

    estudiante = relationship("Estudiante", back_populates="inscripciones")
    curso = relationship("Curso", back_populates="inscripciones")

    __table_args__ = (
        UniqueConstraint("estudiante_id", "curso_id", name="uq_estudiante_curso"),
        CheckConstraint("calificacion >= 0 AND calificacion <= 100", name="chk_calificacion"),
    )


# In[6]:


# -------------------------
# Crear la BD
# -------------------------
if __name__ == "__main__":
    engine = create_engine("sqlite:///universidad.db")
    Base.metadata.create_all(engine)

