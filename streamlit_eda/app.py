from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

COLUMN_EMOJI = {
    "CREDIT": "💳",
    "GEN_TITLE": "👩‍💼",
    "MARITAL_STATUS": "💑",
    "AGE": "👴",
    "SOCSTATUS_PENS_FL": "📜",
    "REG_ADDRESS_PROVINCE": "🏠",
    "DEPENDANTS": "👨‍👩‍👦‍👦",
    "OWN_AUTO": "🚗",
    "GEN_INDUSTRY": "🏭",
    "EDUCATION": "📚",
    "SOCSTATUS_WORK_FL": "👔",
    "JOB_DIR": "💼",
    "TARGET": "🎯",
    "FL_PRESENCE_FL": "🏠",
    "TERM": "📅",
    "AGREEMENT_RK": "📝",
    "CHILD_TOTAL": "👶",
    "POSTAL_ADDRESS_PROVINCE": "✉️",
    "PERSONAL_INCOME": "💰",
    "GENDER": "⚧️",
    "FST_PAYMENT": "💳",
    "WORK_TIME": "⏰",
    "ID_CLIENT": "🆔",
    "FAMILY_INCOME": "👨‍👩‍👦‍👦",
    "FACT_ADDRESS_PROVINCE": "🏠",
    "LOAN_NUM_TOTAL": "📝",
    "LOAN_NUM_CLOSED": "🔒",
}

COLUMNS_DESCRIPTION = """- `CREDIT` — сумма последнего кредита клиента (в рублях).
- `GEN_TITLE` — должность.
- `MARITAL_STATUS` — семейное положение.
- `AGE` — возраст клиента.
- `SOCSTATUS_PENS_FL` — социальный статус клиента относительно пенсии (1 — пенсионер, 0 — не пенсионер).
- `REG_ADDRESS_PROVINCE` — область регистрации клиента.
- `DEPENDANTS` — количество иждивенцев клиента.
- `OWN_AUTO` — количество автомобилей в собственности.
- `GEN_INDUSTRY` — отрасль работы клиента.
- `EDUCATION` — образование.
- `SOCSTATUS_WORK_FL` — социальный статус клиента относительно работы (1 — работает, 0 — не работает).
- `JOB_DIR` — направление деятельности внутри компании.
- `TARGET` — целевая переменная: отклик на маркетинговую кампанию (1 — отклик был зарегистрирован, 0 — отклика не было).
- `FL_PRESENCE_FL` — наличие в собственности квартиры (1 — есть, 0 — нет).
- `TERM` — срок кредита.
- `AGREEMENT_RK` — уникальный идентификатор объекта в выборке.
- `CHILD_TOTAL` — количество детей клиента.
- `POSTAL_ADDRESS_PROVINCE` — почтовый адрес области.
- `PERSONAL_INCOME` — личный доход клиента (в рублях).
- `GENDER` — пол клиента (1 — мужчина, 0 — женщина).
- `FST_PAYMENT` — первоначальный взнос (в рублях).
- `WORK_TIME` — время работы на текущем месте (в месяцах).
- `FAMILY_INCOME` — семейный доход (несколько категорий).
- `ID_CLIENT` — идентификатор клиента.
- `FACT_ADDRESS_PROVINCE` — область фактического пребывания клиента.
- `LOAN_NUM_TOTAL` — количество кредиктов.
- `LOAN_NUM_CLOSED` — количество закрытых кредитов."""

COLUMN_TO_DESCRIPTION = {
    row.split()[1].strip()[1:-1]: row.split(" — ", maxsplit=1)[-1]
    for row in COLUMNS_DESCRIPTION.split("\n")
}

BINARY_COLUMNS = [
    "SOCSTATUS_PENS_FL",
    "SOCSTATUS_WORK_FL",
    "TARGET",
    "FL_PRESENCE_FL",
    "GENDER",
]

CATEGORICAL_COLUMNS = [
    "CHILD_TOTAL",
    "GEN_TITLE",
    "POSTAL_ADDRESS_PROVINCE",
    "MARITAL_STATUS",
    "OWN_AUTO",
    "DEPENDANTS",
    "REG_ADDRESS_PROVINCE",
    "GEN_INDUSTRY",
    "EDUCATION",
    "JOB_DIR",
    "FAMILY_INCOME",
    "FACT_ADDRESS_PROVINCE",
]

ID_COLUMNS = ["AGREEMENT_RK", "WORK_TIME", "ID_CLIENT"]

DATA = data = st.cache_data(pd.read_csv)(
    Path(__name__).parent / "data" / "PREPARED_DATA.csv"
)
NUMERIC_COLUMNS = list(
    set(DATA.columns) - set(BINARY_COLUMNS) - set(CATEGORICAL_COLUMNS) - set(ID_COLUMNS)
)
MODEL = joblib.load(Path(__name__).parent / "models" / "model.pkl")


def get_list_with_desctiption(columns: list[str]) -> str:
    final_str = ""
    for column in columns:
        description = COLUMN_TO_DESCRIPTION.get(column, "")
        final_str += f"* `{column}` — {description}\n"
    return final_str


not_id_columns = [
    column for column in data.columns if "ID" not in column and "RK" not in column
]

st.title("EDA данных о клиентах банка")
st.image("https://managementweekly.org/wp-content/uploads/2021/04/header.jpg")
st.info(
    """
        🚨 Этот дашборд содержит разведочный анализ данных о клиентах банка, включая множество графиков
        по всем имеющимся колонкам, а также возможность построить прогноз для произвольных данных!
"""
)

st.sidebar.title("О дашборде")
st.sidebar.info("EDA данных о клиентах банка")

tab1, tab2 = st.tabs(["EDA", "Предсказание"])

with tab1:
    if st.sidebar.checkbox("⚡Случайная выборка данных", value=True):
        st.subheader("⚡ Случайная выборка строк из таблицы")
        st.write(
            "Посмотрим на то как выглядят данные, для этого возьмём случайные 5 строк."
        )
        if st.button("🔄 Обновить выборку"):
            st.dataframe(data.sample(5), hide_index=True)
        else:
            st.dataframe(data.sample(5), hide_index=True)

        if st.checkbox("Показать описание колонок", value=False):
            st.info(COLUMNS_DESCRIPTION)

    if st.sidebar.checkbox("📈 Статистики для колонок", value=True):
        st.subheader("📈 Статистики для колонок")
        st.write(
            "Теперь выведем статистики для колонок. Рассмотрим все колонки - категориальные и числовые."
        )
        st.dataframe(data.describe(include="all"))
        st.write("`Вывод`: можно выделить определенные группы признаков.")
        st.write("Бинарные:")
        st.write(get_list_with_desctiption(BINARY_COLUMNS))
        st.write("Категориальные:")
        st.write(get_list_with_desctiption(CATEGORICAL_COLUMNS))
        st.write("Числовые:")
        st.write(get_list_with_desctiption(NUMERIC_COLUMNS))
        st.write("Столбцы-идентификаторы:")
        st.write(get_list_with_desctiption(ID_COLUMNS))

    if st.sidebar.checkbox("💑 Попарные scatter графики", value=True):
        st.subheader("💑 Попарные scatter графики")
        st.write(
            "Построим попарные графики, в том числе с целевой переменной, чтобы изучить зависимость."
        )
        col1, col2 = st.columns(2)
        with col1:
            first_column = st.selectbox(
                "Первый столбец", list(DATA.columns), list(DATA.columns).index("TARGET")
            )

        with col2:
            second_column = st.selectbox(
                "Второй столбец",
                list(
                    set(DATA.columns)
                    - {
                        first_column,
                    }
                ),
            )
        fig = px.scatter(
            data,
            x=first_column,
            y=second_column,
            title=f"{first_column} VS {second_column}",
        )
        st.plotly_chart(fig)

    if st.sidebar.checkbox("📶 Распределения для числовых колонок", value=True):
        st.subheader("📶 Распределения для числовых колонок")
        selected_numeric_col = st.selectbox(
            "Выберите столбце для отрисовки гистограммы:", NUMERIC_COLUMNS
        )
        st.write(COLUMN_TO_DESCRIPTION.get(selected_numeric_col, "").capitalize())
        if st.checkbox("Добавить разделение по целевой переменной", value=False):
            kwargs = dict(color="TARGET")
        else:
            kwargs = dict()

        fig = px.histogram(
            data,
            x=selected_numeric_col,
            marginal="box",
            title=f"Гистограмма значений {selected_numeric_col}.",
            **kwargs,
        )
        st.plotly_chart(fig)
        st.write(
            """`Вывод:` однозначно можно сказать, что в данных есть выбросы, 
                 например это прослеживается, если посмотреть на доход - основная вероятностная масса
                 сосредоточена на интервале до 50_000, при этом есть значения большие 200_000. 
                 Данный факт нужно будет учесть при построении модели.
                 В остальном - нельзя сказать, что при разделении распределений по целевой переменной распределения сильно
                 меняются - вероятно это связано с тем, что выборка крайне несбалансированная относительно целевой переменной.
        """
        )

    if st.sidebar.checkbox("🔗 Матрица корреляций", value=True):
        st.subheader("🔗 Матрица корреляций")
        st.info(
            """Построим тепловую карту корреляций для числовых значений.
                 Это может помочь понять связь переменных с целевой переменной, а также между собой."""
        )
        numeric_df = data.loc[:, NUMERIC_COLUMNS + BINARY_COLUMNS].select_dtypes(
            np.number
        )
        corr = numeric_df.corr()
        colors = [
            [0.0, "blue"],
            [0.5, "white"],
            [1.0, "red"],
        ]
        fig = px.imshow(
            corr,
            color_continuous_scale=colors,
            zmin=-1,
            zmax=1,
            title="Тепловая карта для матрицы корреляций.",
        )
        st.plotly_chart(fig)
        st.write(
            """`Вывод:` в целом сильного влияния на целевую переменную не наблюдается, при этом 
                 определенная отрицательная корреляция видна у возраста. Между собой признаки тоже не сильно скоррелированы
                 за исключения янвых примеров (например, понятно, что чем больше кредитов, тем в среднем больше закрытых кредитов).
        """
        )

    if st.sidebar.checkbox(
        "📊 Распределение значений в категориальных и бинарных колонках", value=True
    ):
        st.subheader("📊 Распределение значений в категориальных и бинарных колонках")
        categorical_cols = CATEGORICAL_COLUMNS + BINARY_COLUMNS
        selected_categorical_col = st.selectbox(
            "Select a categorical column", categorical_cols
        )
        st.write(COLUMN_TO_DESCRIPTION.get(selected_categorical_col, "").capitalize())
        df = data[[selected_categorical_col, "TARGET"]].astype(str)
        if st.checkbox("Добавить разделение по целевой переменной ", value=False):
            kwargs = dict(color="TARGET")
        else:
            kwargs = dict()
        fig = px.histogram(
            df,
            x=selected_categorical_col,
            title=f"Распределение значений в {selected_categorical_col}.",
            **kwargs,
        )
        st.plotly_chart(fig)


with tab2:
    st.write(
        """
             На данной вкладке вы можете получить предсказания для любой комбинации параметров,
             Для этого просто заполните все поля и нажмите кнопку `Предсказать`.
    """
    )
    d = {}
    current_columns = []
    for column in sorted(DATA.columns, key=lambda column: column in NUMERIC_COLUMNS):
        if column == "TARGET" or column in ID_COLUMNS:
            continue
        
        if not current_columns:
            current_columns.extend(st.columns(3))
        
        ctcol = current_columns[0]
        current_columns = current_columns[1:]
        
        with ctcol:
            if column in NUMERIC_COLUMNS:
                d[column] = st.slider(
                    label=f"{COLUMN_EMOJI.get(column, '')} {column}",
                    min_value=DATA[column].min(),
                    max_value=DATA[column].max(),
                )
            else:
                d[column] = st.selectbox(
                    f"{COLUMN_EMOJI.get(column, '')} {column}",
                    DATA[column].unique(),
                )

    if st.button("Предсказать!"):
        data_to_predict = pd.DataFrame([d])
        predict = MODEL.predict(data_to_predict)[0]
        predict_proba = MODEL.predict_proba(data_to_predict)[0, predict]
        st.success(
            f"""
            ✅ Предсказание успешное построено!
            Итоговое предсказаний - {predict} с вероятностью {predict_proba}.
            """
        )
