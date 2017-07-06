(ns russia-housing.core
  (:require [clojure.java.io :as io]
            [clojure.string :as string]
            [clojure.data.csv :as csv]
            [cortex.nn.layers :as layers]
            [cortex.nn.network :as network]
            [cortex.optimize.adam :as adam]
            [cortex.metrics :as metrics]
            [cortex.util :as util]
            [cortex.experiment.train :as experiment-train]
            [clj-time.core :as t]
            [clj-time.format :as f]))

(def data-file "data/housing_data.csv") ;30471

(def params
  {:test-size   1920
   :optimizer   (adam/adam)
   :batch-size  128
   :epoch-size  1024})

(defn csv->maps
  "Turn csv data into maps and try to read-string the values"
  [csv-data]
  (map zipmap
       (->> (first csv-data) ;; First row is the header
            (map keyword) ;; Drop if you want string keys instead
            repeat)
       (->> (rest csv-data)
            (map (fn [row]
                   (map #(try
                           (read-string %)
                           (catch Exception e
                             %))
                        row))))))

(defn write-csv [data-map path]
  (let [columns (keys (first data-map))
        headers (map name columns)
        rows (mapv #(mapv % columns) data-map)]
    (with-open [file (io/writer path)]
      (csv/write-csv file (cons headers rows)))))


(defn convert-date
  "Given a base day and a current day in 'year-month-day' string format,
  return the number of days between the two."
  [base-day current-day]
  (let [formatter (f/formatters :year-month-day)
        base (f/parse formatter base-day)
        current (f/parse formatter current-day)]
    (t/in-days (t/interval base current))))


(defn label->one-hot
  "Given a vector of class-names and a label, return a one-hot vector based on
  the position in class-names.
  E.g.  (label->vec [:a :b :c :d] :b) => [0 1 0 0]"
  [class-names label]
  (let [num-classes (count class-names)
        src-vec (vec (repeat num-classes 0))
        label-idx (.indexOf class-names label)]
    (when (= -1 label-idx)
      (throw (ex-info "Label not in classes for label->one-hot"
                      {:class-names class-names :label label})))
    (assoc src-vec label-idx 1)))


(defn one-hot-encoding
  "Given a dataset and a list of categorical features, returns a new dataset with these
  features encoded into one-hot indicators

  E.g. (one-hot-encoding [{:a :left} {:a :right} {:a :top}] [:a])
         => [{:a_0 1 :a_1 0 :a_2 0} {:a_0 0 :a_1 1 :a_2 0} {:a_0: 0 :a_1 0 :a_2 1}]"
  [dataset features]
  (reduce (fn [mapseq key]
            (let [classes (set (map key mapseq))
                  new-keys (for [i (range (count classes))]
                             (keyword (str (name key) "_" i)))] ; a_0, a_1, a_2, etc.
              (map (fn [elem] (->> (label->one-hot (vec classes) (key elem))
                                   (zipmap new-keys)
                                   (merge (dissoc elem key))))
                   mapseq)))
          dataset features))

(defn convert-numerical
  "Given a dataset, a list of features to convert, and a value to match,
  return the  dataset with those features mapped to 1 if the old value matches the given value
  and 0 otherwise
  E.g. map yes/no to 1/0"
  [dataset features positive]
  (reduce (fn [mapseq k]
            (map #(update % k (fn [val] (if (= val positive) 1 0))) mapseq))
          dataset features))


(defn dataset->feature-map
  "Given a dataset (seq of maps, one per data point),
  return a map of {:feature_name [data for this feature]}"
  [dataset]
  (let [feat-list (keys (first dataset))
        feat-vals (mapv (fn [feature]
                          (mapv feature dataset)) feat-list)]
    (zipmap feat-list feat-vals)))


(defn- get-means
  "Given a dataset, return a map of each feature to its mean value (ignoring NA's)"
  [feature-map]
  (let [feat-names (keys feature-map)
        filled-feats (map (fn [elem] (filter #(not= 'NA %) elem)) (vals feature-map))
        means (map #(/ (reduce + %) (count %)) filled-feats)]
    (zipmap feat-names means)))


(defn fill-impute
  "Given a dataset, fill missing values ('NA) with the mean value of the feature"
  [dataset]
  (let [feature-map (dataset->feature-map dataset)
        means (get-means feature-map)
        ;; for each feature, map over each list of values, if NA then fill with mean
        imputed-feat-vals (for [feature feature-map
                                :let [feat-name (key feature)
                                      feat-vals (val feature)]]
                            (mapv (fn [value]
                                    (if (= 'NA value)
                                      (feat-name means)
                                      value)) feat-vals))]
    (partition (count imputed-feat-vals) (apply interleave imputed-feat-vals))))


(def dataset
  (future
    (let [csv-data (with-open [infile (io/reader data-file)]
                     (csv->maps (doall (csv/read-csv infile))))
          base-day (:timestamp (first csv-data))
          conv-data (as-> csv-data d
                      (map #(dissoc % :id) d) ; drop id column
                      (map #(assoc % :timestamp (convert-date base-day (:timestamp %))) d)
                      (one-hot-encoding d [:sub_area :ecology])
                      (convert-numerical d [:product_type] 'Investment)
                      (convert-numerical d [:culture_objects_top_25 :thermal_power_plant_raion :water_1line
                                            :incineration_raion :oil_chemistry_raion :radiation_raion
                                            :railroad_terminal_raion :big_market_raion :nuclear_reactor_raion
                                            :detention_facility_raion :big_road1_1line :railroad_1line] 'yes))
          labels (map :price_doc conv-data)
          imputed-data (fill-impute (map #(dissoc % :price_doc) conv-data))]
      (mapv (fn [d l] {:data d :label l}) imputed-data labels))))



(defn infinite-dataset
  "Given a finite dataset, generate an infinite sequence of maps partitioned
  by :epoch-size"
  [map-seq & {:keys [epoch-size]
              :or {epoch-size 1024}}]
  (->> (repeatedly #(shuffle map-seq))
       (mapcat identity)
       (partition epoch-size)))


(def network-description
  [(layers/input (count (:data (first @dataset))) 1 1 :id :data)
   (layers/linear->relu 512)
   (layers/dropout 0.9)
   (layers/linear->relu 256)
   (layers/dropout 0.9)
   (layers/linear->relu 128)
   (layers/linear->relu 32)
   (layers/dropout 0.8)
   (layers/linear->relu 8)
   (layers/linear 1 :id :label)])


(defn train
  "Trains network for :epoch-count number of epochs"
  []
  (println network-description)
  (let [network (network/linear-network network-description)
        [train-ds test-ds] [(infinite-dataset (drop (:test-size params) @dataset))
                            (take (:test-size params) @dataset)]]
    (experiment-train/train-n network train-ds test-ds
                              :batch-size (:batch-size params)
                              :epoch-count (:epoch-count params)
                              :optimizer (:optimizer params))))
