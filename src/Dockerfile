FROM python:alpine3.20

# curl is needed for the poetry install
RUN apk add curl

#RUN apk add py3-scikit-learn

# gcc et. al. is needed for the scikit-learn install
# https://stackoverflow.com/questions/56590386/is-sklearn-compatible-with-linux-alpine
RUN apk add gcc g++ gfortran lapack-dev libffi-dev libressl-dev musl-dev openblas-dev

# Setup app user
ENV USER="app"
ENV HOME_DIR="/home/app"

# Create an app user so we don't have to be root
RUN addgroup --gid 1001 -S $USER && adduser -G $USER --shell /bin/false --disabled-password -H --uid 1001 $USER
USER $USER
WORKDIR $HOME_DIR

# Setup Poetry
# NOTE: Don't rely on external dependencies in "real" world situations...
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$HOME_DIR/.local/bin:$PATH"

# Install package dependencies
ADD poetry.lock pyproject.toml $HOME_DIR
RUN poetry install

ENTRYPOINT [ "/bin/sh" ]
