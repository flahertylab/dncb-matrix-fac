FROM python:3.9

RUN apt-get update && apt-get install -y \
    build-essential \
    libgsl-dev \
    vim

WORKDIR /work

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the module code and setup.py
COPY src /work/src

# Install the dncb-fac module
WORKDIR /work/src
RUN python setup.py build_ext
RUN pip install .

WORKDIR /work/

# Set the entry point
CMD ["dncbfac"]