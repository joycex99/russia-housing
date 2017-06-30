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
  "Turn csv data into maps"
  [csv-data]
  (map zipmap
       (->> (first csv-data) ;; First row is the header
            (map keyword) ;; Drop if you want string keys instead
            repeat)
       (rest csv-data)))


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
      (throw (ex-info "Label not in class for label->one-hot"
                      {:class-names class-names :label label})))
    (assoc src-vec label-idx 1)))


(defn mapseq->categorical
  "Given a map sequence and a common key with categorical data, return a map sequence with
  one-hot indicator values for that key"
  [mapseq key]
  (let [classes (set (map key mapseq))
        new-keys (for [i (range (count classes))]
                   (keyword (str (name key) "_" i)))] ; a_0, a_1, a_2, etc.
    (map (fn [elem] (->> (label->one-hot (vec classes) (key elem))
                         (zipmap new-keys)
                         (merge (dissoc elem key))))
         mapseq)))


(def dataset
  (future
    (let [csv-data (with-open [infile (io/reader data-file)]
                     (csv->maps (doall (csv/read-csv infile))))
          base-day (:timestamp (first csv-data))
          int-data (->> csv-data
                        (map #(dissoc % :id)) ; drop id column
                        (map #(assoc % :timestamp (convert-date base-day (:timestamp %))))
                        )
          ]
      int-data)))

                                        ; (def mapdata (with-open [infile (io/reader data-file)] (csv-data->maps (doall (csv/read-csv infile)))))

;; (def dataset
;;   (future
;;     (let [ind-data (with-open [infile (io/reader data-file)]
;;                      (rest (doall (csv/read-csv infile))))
;;           data (->> ind-data
;;                     (map rest)                ; drop first col (label)
;;                     (map #(map read-string %)))
;;           labels (->> ind-data
;;                       (map first)
;;                       (map read-string))]
;;       (mapv (fn [d l] {:data d :label l}) data labels))))

;; (defn infinite-dataset
;;   "Given a finite dataset, generate an infinite sequence of maps partitioned
;;   by :epoch-size"
;;   [map-seq & {:keys [epoch-size]
;;               :or {epoch-size 1024}}]
;;   (->> (repeatedly #(shuffle map-seq))
;;        (mapcat identity)
;;        (partition epoch-size)))


;; (def network-description
;;   [(layers/input (count (:data (first @dataset))) 1 1 :id :data)
;;    (layers/linear->relu 8)
;;    (layers/linear 1 :id :label)])

;; (defn train
;;   "Trains network for :epoch-count number of epochs"
;;   []
;;   (let [network (network/linear-network network-description)
;;         [train-ds test-ds] [(infinite-dataset (drop (:test-size params) @dataset))
;;                             (take (:test-size params) @dataset)]]
;;     (experiment-train/train-n network train-ds test-ds
;;                               :batch-size (:batch-size params)
;;                               :epoch-count (:epoch-count params)
;;                               :optimizer (:optimizer params))))
